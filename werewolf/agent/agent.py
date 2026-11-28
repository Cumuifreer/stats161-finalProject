"""狼人殺 agent v1

架構：官方的 LLMAgent (examples/llm_agent.py) 負責跟 LLM 溝通，LLM 用 Claude (claude_provider.py)，
這裡多加一個 Notebook 在程式端記帳 (角色、查驗、死者身份、投票、誰跳預言家)，
算一個簡單的狼人機率，每次決策前塞進 prompt。最後決定還是 LLM 做。

Pokémon 之類的其他遊戲直接走官方 LLMAgent。
"""

from __future__ import annotations

import json
import os
import re
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

from agent.claude_provider import claude_from_env
from agent.deepseek_provider import deepseek_from_env
from examples.llm.providers import provider_from_env
from altruagent.runner import WithReasoning
from examples.llm_agent import LLMAgent, _accepts_timeout

HERE = Path(__file__).parent
STRATEGY = (HERE / "werewolf_strategy.md").read_text(encoding="utf-8")
LESSONS_FILE = HERE / "lessons.md"          # scripts/reflect.py 寫的賽後檢討
LOG_DIR = HERE.parent / "logs"              # 每局的決策紀錄，報告分析用 (已加進 .gitignore)


ROLE_SECTIONS = {"Villager", "Seer", "Wolf"}


def strategy_for(role, has_opponents=True):
    """只給自己身份那一段策略，省 token。身份還不知道就全給。"""
    parts = re.split(r"(?m)^(?=## )", STRATEGY)
    keep = []
    for p in parts:
        title = p[3:].split("\n", 1)[0].strip() if p.startswith("## ") else ""
        if title in ROLE_SECTIONS and role and title.lower() != role:
            continue
        if title == "Opponents" and not has_opponents:
            continue
        keep.append(p)
    return "".join(keep).strip()


def _lessons():
    try:
        return LESSONS_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
ABSTAIN = 7

# 概似比：某個證據在「這組人是狼」底下，比起不是狼時多常出現
LR = {
    "wolf_votes_ally": 0.2,        # 狼很少投隊友
    "good_votes_wolf": 1.3,        # 好人投到狼的機率比亂投高一點
    "wolf_pushes_mislynch": 2.0,   # 投死好人的人比較像狼
    "victim_suspected_wolf": 1.5,  # 被刀的人生前投的人比較像狼
    "fake_seer": 3.0,              # 我是預言家時，別人跳預言家
    "two_seers_no_wolf": 0.2,      # 兩人跳預言家卻都不是狼，不太可能
    "injector": 1.5,               # 發假系統訊息
    "self_reveal_wolf": 10.0,      # 自己說是狼
    "wolf_accuses_ally": 0.5,      # 狼很少在聊天指控隊友
    "accused_good": 1.3,           # 指控過後來翻牌是好人的人
    "accused_me": 1.3,             # 我知道自己是好人，指控我的人
    "voted_me": 1.5,               # 我知道自己是好人，投我的人比較像狼
    "voted_claimed_seer_me": 2.5,  # 我是預言家而且跳了，還投我
}


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
        self.injectors = set()    # 在聊天裡假裝系統/下指令的人
        self.self_wolf = set()    # 自己承認是狼的人
        self.accuse = set()       # (說話的人, 被他指控的人)，從聊天抓
        self.dm_days = set()      # 我有發私訊的天
        self._read = 0
        self.dms_to_me = []       # 別人私訊我的內容
        self.pairs = []           # 最可能的狼組合 (貝氏算出來的前 3 名)

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

        new, self._read = transcript[self._read:], len(transcript)   # 每則訊息只看一次，才知道是哪天講的
        for msg in new:
            if re.search(r"\b(i am|i'm)\b[^.]*\bseer\b", msg["text"], re.I):
                self.seer_claims.add(_seat_from_label(msg["from"]))
            if re.search(r"\b(i am|i'm)\s+(a |the )?(were)?wolf\b", msg["text"], re.I):
                self.self_wolf.add(_seat_from_label(msg["from"]))
            src = _seat_from_label(msg["from"])
            for g in ACCUSE.findall(msg["text"]):
                t = int(g[0] or g[1])
                if src is not None and t != src:
                    self.accuse.add((src, t))
            to = msg.get("to")
            if isinstance(to, list) and to:          # 私訊
                if src == self.me:
                    self.dm_days.add(self.day)
                elif f"Player{self.me}" in to:
                    self.dms_to_me.append({"day": self.day, "from": src, "text": msg["text"]})
            if INJECTION.search(msg["text"]):
                self.injectors.add(_seat_from_label(msg["from"]))

    def wolf_probs(self):
        """每個活著的其他人是狼的機率 (貝氏)。

        7 人局只有 2 隻狼，把「哪兩個人是狼」的所有組合 (最多 15 種) 列出來，
        先用確定的事實刪掉不可能的組合 (死者翻牌、預言家查驗)，
        剩下的組合各自乘上觀察到的證據的概似比 (LR 表)，最後加總得到每個人的後驗機率。
        """
        self.pairs = []
        others = [s for s in self.alive if s != self.me]
        if self.role == "wolf":     # 狼自己知道答案
            return dict(sorted({s: float(s in self.allies) for s in others}.items(), key=lambda kv: -kv[1]))

        dead_role = {s: r for s, (r, _, _) in self.dead.items()}
        cands = [s for s in range(7) if s != self.me]
        post = {}
        for pair in combinations(cands, 2):
            P = set(pair)
            if any((r == "wolf") != (s in P) for s, r in dead_role.items() if r):
                continue
            if any((r == "wolf") != (s in P) for s, r in self.checks.items()):
                continue
            post[pair] = self._likelihood(P)
        total = sum(post.values())
        if not total:               # 證據互相矛盾 (有人說謊或解析錯)，退回平均
            post = {pr: 1.0 for pr in combinations(cands, 2)}
            total = len(post)
        probs = {s: sum(w for pr, w in post.items() if s in pr) / total for s in others}
        self.pairs = sorted(((list(pr), round(w / total, 3)) for pr, w in post.items()), key=lambda x: -x[1])[:3]
        return dict(sorted(probs.items(), key=lambda kv: -kv[1]))

    def _likelihood(self, P):
        """這組狼 P 底下，看到目前所有證據的相對機率。LR 是手調的，之後可以用對局 log 估。"""
        L = 1.0
        for _, v, t in self.votes:
            if v is None or t is None:
                continue
            if t == self.me and v in P:
                L *= LR["voted_claimed_seer_me"] if self.role == "seer" and self.me in self.seer_claims else LR["voted_me"]
            if v in P and t in P:
                L *= LR["wolf_votes_ally"]
            elif v not in P and t in P:
                L *= LR["good_votes_wolf"]
            if t in self.dead and self.dead[t][1] == "vote" and self.dead[t][0] != "wolf" and v in P:
                L *= LR["wolf_pushes_mislynch"]
            if v in self.dead and self.dead[v][1] == "night_kill" and t in P:
                L *= LR["victim_suspected_wolf"]
        for a, t in self.accuse:
            if a in P and t in P:
                L *= LR["wolf_accuses_ally"]
            if a in self.dead and self.dead[a][1] == "night_kill" and t in P:
                L *= LR["victim_suspected_wolf"]
            if t in self.dead and self.dead[t][0] != "wolf" and a in P:
                L *= LR["accused_good"]
            if t == self.me and a in P:
                L *= LR["accused_me"]
        claims = self.seer_claims - {None}
        if self.role == "seer":
            L *= LR["fake_seer"] ** len(claims & P)
        elif len(claims) > 1 and not claims & P:
            L *= LR["two_seers_no_wolf"]
        L *= LR["injector"] ** len(self.injectors & P)
        L *= LR["self_reveal_wolf"] ** len(self.self_wolf & P)
        return L

    def seer_suspects(self):
        """看起來像預言家的人：有跳的、或投票一直打到狼的。狼晚上刀人參考。"""
        hits = {}
        for _, voter, target in self.votes:
            if target in self.dead and self.dead[target][0] == "wolf":
                hits[voter] = hits.get(voter, 0) + 1
        for a, t in self.accuse:            # 指控我或隊友的人，可能有查驗資訊
            if t in self.allies | {self.me}:
                hits[a] = hits.get(a, 0) + 1
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
        # 好幾個座位同名 (例如官方 Reference Easy) 就分不出誰是誰，不記
        dup = {n for n in self.names.values() if list(self.names.values()).count(n) > 1}
        return {n: r for n, r in out.items() if n not in dup}

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
            "likely_wolf_pairs": self.pairs or None,
            "votes_by_day": _votes_by_day(self.votes),
            "accusations": sorted([a, t] for a, t in self.accuse) or None,
            "private_messages_to_me": self.dms_to_me[-6:] or None,
            "sent_private_message_today": self.day in self.dm_days,
            "fake_system_messages_from": sorted(s for s in self.injectors if s is not None) or None,
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
            extra = f"\n\nStrategy:\n{strategy_for(self.notes.get('my_role'), bool(self.notes.get('opponents')))}"
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
        legal = list(state.legal_actions or [])
        if context.game_type == "werewolf" and len(legal) == 1:
            return WithReasoning(legal[0], "Only one option.")  # 只有一個選擇就不問 LLM
        out = super().choose_action(state, context)
        if context.game_type == "werewolf" and isinstance(out, WithReasoning) and LEAK.search(out.reasoning_summary or ""):
            out = WithReasoning(out.action, "Decided from the game record.")  # 公開理由不能洩漏身份
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

# 聊天裡的指控："suspect Player3"、"vote Player3"、"Player3 is a wolf"
ACCUSE = re.compile(r"(?:suspect|vote|voting|accuse)\w*\s+(?:on\s+)?Player(\d+)|Player(\d+)\s+is\s+(?:a\s+)?(?:were)?wolf", re.I)

# 公開理由裡會洩漏身份的字
LEAK = re.compile(r"\b(seer|checked|check|investigat\w*|my ally|fellow wolf|partner|teammate)\b", re.I)

# 假裝是系統、叫人公布身份/照做的訊息
INJECTION = re.compile(
    r"\b(system|admin|moderator|host|game master)\b\s*[:\]]|ignore (all|previous|your)|"
    r"new instructions?|you must (reveal|tell|say|vote)|all (players|agents) must|reveal your (true )?role",
    re.I)


def _votes_by_day(votes):
    out = {}
    for day, v, t in votes:
        if v is not None:
            out.setdefault(str(day), {})[v] = t if t is not None else "abstain"
    return out


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
    """DRY_RUN=1 就不用 LLM；再來依序看 Claude、DeepSeek，都沒有就用 OpenAI。"""
    if os.environ.get("DRY_RUN") == "1":
        return DryRunProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return claude_from_env()
    if os.environ.get("DEEPSEEK_API_KEY"):
        return deepseek_from_env()
    return provider_from_env()


def create_agent():
    load_dotenv(override=True)  # .env 優先，免得終端機裡舊的 key 蓋過去
    return MyAgent(pick_provider())
