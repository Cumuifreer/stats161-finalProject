"""狼人殺 agent v1

架構：官方的 LLMAgent (examples/llm_agent.py) 負責跟 LLM 溝通，LLM 用 Claude (claude_provider.py)，
這裡多加一個 Notebook 在程式端記帳 (角色、查驗、死者身份、投票、誰跳預言家)，
算一個簡單的狼人機率，每次決策前塞進 prompt。最後決定還是 LLM 做。

Pokémon 之類的其他遊戲直接走官方 LLMAgent。
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from agent.claude_provider import claude_from_env
from examples.llm.providers import provider_from_env
from examples.llm_agent import LLMAgent, _accepts_timeout

HERE = Path(__file__).parent
STRATEGY = (HERE / "werewolf_strategy.md").read_text(encoding="utf-8")
LESSONS_FILE = HERE / "lessons.md"          # scripts/reflect.py 寫的賽後檢討
LOG_DIR = HERE.parent / "logs"              # 每局的決策紀錄，報告分析用 (已加進 .gitignore)


def _lessons():
    try:
        return LESSONS_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
ABSTAIN = 7


class Notebook:
    """程式端的記憶，只記事實和算機率，不做決定。"""

    def __init__(self):
        self.me = None
        self.role = None
        self.allies = set()
        self.checks = {}          # seat -> "wolf" / "villager" (只有預言家有)
        self.dead = {}            # seat -> (role, cause, day)
        self.votes = []           # (day, voter_seat, target_seat)
        self.seer_claims = set()  # 在聊天裡自稱預言家的人
        self.alive = set(range(7))
        self.day = 0
        self.names = {}           # seat -> agent 名字，跨場認人用

    def update(self, state, transcript):
        raw = state.raw
        players = raw.get("players") or []
        name2seat = {p["name"]: p["position"] for p in players if isinstance(p, dict)}
        # 跨場認人：名字長得像 "Simon Liu (Player0, Self-hosted)"，座位號每場不同，要拿掉
        self.names = {p["position"]: re.sub(r"\s*\(Player\d+[^)]*\)", "", p["name"])
                      for p in players if isinstance(p, dict)}
        seat = lambda name: name2seat.get(name, _seat_from_label(name))

        self.me = raw.get("your_position", self.me)
        obs = state.observation or ""
        if m := re.search(r"Your role:\s*(\w+)", obs):
            self.role = m.group(1).lower()
        if m := re.search(r"fellow wolves?:\s*([^\n]*)", obs, re.I):
            self.allies = {int(x) for x in re.findall(r"Player(\d+)", m.group(1))}
        for s, res in re.findall(r"Player(\d+) is (wolf|villager)", obs):
            self.checks[int(s)] = res

        gs = raw.get("game_state") or {}
        self.day = gs.get("day") or self.day
        if gs.get("alive"):
            self.alive = {s for s in map(seat, gs["alive"]) if s is not None}
        for d in gs.get("dead") or []:
            s = seat(d.get("player"))
            if s is not None:
                self.dead[s] = (d.get("role"), d.get("cause"), d.get("day"))
        self.votes = [(day, seat(v), seat(t)) for day, v, t in _iter_votes(gs.get("vote_history"))]

        for msg in transcript:
            if re.search(r"\b(i am|i'm)\b[^.]*\bseer\b", msg["text"], re.I):
                self.seer_claims.add(_seat_from_label(msg["from"]))

    def wolf_probs(self):
        """每個活著的其他人是狼的機率。

        規則：查驗結果和隊友直接定案；其他人用投票紀錄打分
        (投給後來被揭露是好人的 +1，投給狼的 -1)，再把剩下的狼數按 softmax 分配。
        """
        wolves_dead = sum(1 for r, _, _ in self.dead.values() if r == "wolf")
        known_wolves = {s for s, r in self.checks.items() if r == "wolf" and s in self.alive}
        if self.role == "wolf":
            known_wolves |= self.allies & self.alive
        good = {s for s, r in self.checks.items() if r == "villager"}
        others = [s for s in self.alive if s != self.me]
        unknown = [s for s in others if s not in known_wolves and s not in good]
        left = max(0, 2 - wolves_dead - len(known_wolves) - (self.role == "wolf"))

        score = {s: 0.0 for s in unknown}
        for _, voter, target in self.votes:
            if voter in score and target in self.dead:
                score[voter] += -1.0 if self.dead[target][0] == "wolf" else 1.0
        if len(self.seer_claims) > 1:           # 兩個人跳預言家，至少一個是狼
            for s in self.seer_claims & set(score):
                score[s] += 1.0

        w = {s: math.exp(v) for s, v in score.items()}
        total = sum(w.values()) or 1
        probs = {s: min(1.0, left * w[s] / total) for s in unknown}
        probs.update({s: 1.0 for s in known_wolves if s != self.me})
        probs.update({s: 0.0 for s in good if s in others})
        return dict(sorted(probs.items(), key=lambda kv: -kv[1]))

    def seer_suspects(self):
        """看起來像預言家的人：有跳的、或投票一直打到狼的。狼晚上刀人參考。"""
        hits = {}
        for _, voter, target in self.votes:
            if target in self.dead and self.dead[target][0] == "wolf":
                hits[voter] = hits.get(voter, 0) + 1
        sus = (self.seer_claims | set(hits)) & self.alive - self.allies - {self.me, None}
        return sorted(sus, key=lambda s: (s not in self.seer_claims, -hits.get(s, 0)))

    def this_game(self):
        """這一場每個對手的表現，給 Opponents 存檔。"""
        out = {}
        for s, name in self.names.items():
            if s == self.me:
                continue
            row = {"role": self.dead.get(s, (None,))[0], "claimed_seer": s in self.seer_claims,
                   "hit_wolf": 0, "hit_good": 0}
            for _, voter, target in self.votes:
                if voter == s and target in self.dead:
                    row["hit_wolf" if self.dead[target][0] == "wolf" else "hit_good"] += 1
            out[name] = row
        return out

    def summary(self, opponents=None):
        wolves_dead = sum(1 for r, _, _ in self.dead.values() if r == "wolf")
        return {
            "me": self.me, "my_role": self.role, "day": self.day,
            "my_allies": sorted(self.allies) if self.role == "wolf" else None,
            "my_seer_checks": self.checks or None,
            "alive": sorted(self.alive),
            "dead": {s: {"role": r, "cause": c, "day": d} for s, (r, c, d) in self.dead.items()},
            "wolves_left": 2 - wolves_dead,
            "seer_claims": sorted(s for s in self.seer_claims if s is not None),
            "wolf_prob": {s: round(p, 2) for s, p in self.wolf_probs().items()},
            "seer_suspects": self.seer_suspects() if self.role == "wolf" else None,
            "opponents": {s: opponents.get(self.names.get(s)) for s in self.alive
                          if s != self.me and opponents and opponents.get(self.names.get(s))} or None,
        }


class Opponents:
    """跨場的對手紀錄。檔案存 {名字: {session_id: 那一場的表現}}，
    同一場重複寫只會覆蓋，不會重複算。"""

    def __init__(self, path=None):
        self.path = path or LOG_DIR / "opponents.json"
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def save_game(self, session_id, game):
        if not game:
            return
        for name, row in game.items():
            self.data.setdefault(name, {})[str(session_id)] = row
        try:
            self.path.parent.mkdir(exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def stats(self, skip_session=None):
        """每個對手過去幾場的總結 (不含這一場)。"""
        out = {}
        for name, games in self.data.items():
            past = [g for sid, g in games.items() if sid != str(skip_session)]
            if not past:
                continue
            known = [g for g in past if g["role"]]
            out[name] = {
                "games": len(past),
                "wolf_games": sum(g["role"] == "wolf" for g in known),
                "fake_seer": sum(g["claimed_seer"] and g["role"] not in (None, "seer") for g in past),
                "votes_hit_wolf": sum(g["hit_wolf"] for g in past),
                "votes_hit_good": sum(g["hit_good"] for g in past),
            }
        return out


class NotesProvider:
    """包住 LLM provider，狼人殺的時候把策略和筆記加到 system prompt。"""

    def __init__(self, inner):
        self._inner = inner
        self._takes_timeout = _accepts_timeout(inner)
        self.notes = None
        self.model = getattr(inner, "model", "")

    def complete_structured(self, messages, name, schema, **kw):
        if self.notes is not None:
            extra = f"\n\nStrategy:\n{STRATEGY}"
            if lessons := _lessons():
                extra += f"\n\nLessons from your past games:\n{lessons}"
            extra += "\n\nYour notebook (computed from the game record, trust it):\n" + json.dumps(self.notes)
            messages = [{**messages[0], "content": messages[0]["content"] + extra}, *messages[1:]]
        if not self._takes_timeout:
            kw.pop("timeout", None)
        return self._inner.complete_structured(messages, name, schema, **kw)


class MyAgent(LLMAgent):
    def __init__(self, provider, **kw):
        self.notes_provider = NotesProvider(provider)
        super().__init__(self.notes_provider, **kw)
        self.book = Notebook()
        self.opps = Opponents()

    def _prep(self, state, context):
        if context.game_type == "werewolf":
            self._remember(state)
            self.book.update(state, self._transcript)
            self.opps.save_game(context.session_id, self.book.this_game())
            self.notes_provider.notes = self.book.summary(self.opps.stats(context.session_id))
        else:
            self.notes_provider.notes = None

    def choose_action(self, state, context):
        self._prep(state, context)
        out = super().choose_action(state, context)
        self._record(context, "action", state, getattr(getattr(out, "action", out), "action_id", str(out)),
                  getattr(out, "reasoning_summary", None))
        return out

    def choose_message(self, state, context):
        self._prep(state, context)
        out = super().choose_message(state, context)
        self._record(context, "message", state, getattr(out, "content", "END"), getattr(out, "recipients", None))
        return out

    def _record(self, context, kind, state, choice, extra):
        """一行一筆 JSON，之後拿來算勝率、投票準確度、做 reflection。"""
        if context.game_type != "werewolf":
            return
        try:
            LOG_DIR.mkdir(exist_ok=True)
            row = {"seat": context.seat_position, "kind": kind,
                   "sub_phase": (state.raw.get("game_state") or {}).get("phase"),
                   "choice": choice, "extra": extra, "notes": self.notes_provider.notes,
                   "messages": self._transcript[-10:],
                   "game_state": state.raw.get("game_state")}  # 原始資料，確認 vote_history 格式用
            with open(LOG_DIR / f"{context.session_id}_seat{context.seat_position}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # 寫 log 失敗不能影響比賽


# ---- 小工具 ----

def _seat_from_label(name):
    m = re.search(r"Player(\d+)", str(name or ""))
    return int(m.group(1)) if m else None


def _iter_votes(vh):
    """vote_history 的格式文件沒寫死，幾種常見的都吃，看不懂就跳過 (LLM 還是看得到原始資料)。"""
    if isinstance(vh, dict):
        vh = [{"day": k, "votes": v} for k, v in vh.items()]
    for item in vh or []:
        if not isinstance(item, dict):
            continue
        day = item.get("day")
        if "voter" in item:
            yield day, item.get("voter"), item.get("target")
        votes = item.get("votes")
        if isinstance(votes, dict):
            for v, t in votes.items():
                yield day, v, t
        elif isinstance(votes, list):
            for x in votes:
                if isinstance(x, dict):
                    yield day, x.get("voter"), x.get("target")


class DryRunProvider:
    """不呼叫 LLM 的測試模式 (.env 寫 DRY_RUN=1)：永遠選第一個合法動作、不發言。
    只用來確認連線和收集真實的 log，正式比賽不能用 (課程規定 LLM 要參與決策)。"""

    model = "dry-run"

    def complete_structured(self, messages, schema_name, schema, **kw):
        props = schema.get("properties", {})
        if "action_id" in props:
            return {"action_id": props["action_id"]["enum"][0], "reasoning_summary": "dry run"}
        return {"decision": "end", "message": "", "recipient": None}


def pick_provider():
    """DRY_RUN=1 就不用 LLM；有 ANTHROPIC_API_KEY 就用 Claude，不然用 OpenAI。"""
    if os.environ.get("DRY_RUN") == "1":
        return DryRunProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return claude_from_env()
    return provider_from_env()


def create_agent():
    load_dotenv()
    return MyAgent(pick_provider())
