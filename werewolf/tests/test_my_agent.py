"""用假 LLM 跑：python -m pytest tests/test_my_agent.py -q"""

import pytest
from altruagent import DecisionContext, GameState
from agent.agent import MyAgent, Notebook, Opponents


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    import agent.agent as A
    monkeypatch.setattr(A, "LOG_DIR", tmp_path)  # 測試不要寫到真的 logs/

CTX = DecisionContext(session_id="w", tournament_id=None, game_type="werewolf", agent_id="a2", seat_position=2)
PLAYERS = [{"position": i, "name": f"Bot{i}", "agent_id": f"a{i}"} for i in range(7)]


class FakeLLM:
    model = "fake"

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def complete_structured(self, messages, name, schema, timeout=None):
        self.calls.append(messages)
        return self.answer


def state(obs, legal=(), dead=(), votes=None, alive=range(7), sub="day_vote", phase="moving", msgs=()):
    raw = {
        "session_id": "w", "game_type": "werewolf", "status": "in_progress", "state_version": 5,
        "phase": phase, "observation": obs, "is_terminal": False, "is_current_actor": True,
        "your_position": 2, "players": PLAYERS, "eliminated": False,
        "game_state": {"day": 2, "phase": sub, "alive": [f"Bot{i}" for i in alive], "dead": list(dead),
                       "vote_history": votes},
        "new_messages": [{"seq": i, "sender": s, "recipients": [], "content": t, "type": "chat"} for i, (s, t) in enumerate(msgs)],
        "legal_actions": {"session_id": "w", "state_version": 5, "actions": [
            {"action_id": a, "label": f"Vote Player{a}", "input": {"session_id": "w", "action_id": a, "state_version": 5}}
            for a in legal]},
    }
    return GameState.from_mcp_state(raw)


def test_notebook_seer_and_votes():
    s = state("Your role: SEER\nYour investigations: Player4 is wolf, Player5 is villager",
              dead=[{"day": 1, "player": "Bot1", "role": "villager", "cause": "vote"}],
              votes=[{"day": 1, "votes": {"Bot3": "Bot1", "Bot6": "Bot1", "Bot0": "Bot4"}}],
              alive=[0, 2, 3, 4, 5, 6])
    b = Notebook()
    b.update(s, [])
    p = b.wolf_probs()
    assert b.role == "seer" and p[4] == 1.0 and p[5] == 0.0
    assert p[3] > p[0]  # 投死村民的人比較可疑
    assert abs(sum(v for k, v in p.items() if k != 4) - 1.0) < 1e-6  # 還剩一隻狼沒找到


def test_wolf_knows_ally():
    b = Notebook()
    b.update(state("Your role: WOLF\nYour fellow wolves: Player5"), [])
    assert b.allies == {5} and b.wolf_probs()[5] == 1.0


def test_double_seer_claim():
    b = Notebook()
    b.update(state("Your role: VILLAGER"), [{"from": "Player3", "text": "I am the seer, 4 is wolf"},
                                            {"from": "Player6", "text": "No, I'm the seer."}])
    assert b.seer_claims == {3, 6}


def test_agent_puts_notes_in_prompt():
    llm = FakeLLM({"action_id": "4", "reasoning_summary": "Player4 looks bad."})
    agent = MyAgent(llm, log=lambda *_: None)
    out = agent.choose_action(state("Your role: VILLAGER", legal=["0", "4", "7"]), CTX)
    assert out.action.action_id == "4"
    system = llm.calls[0][0]["content"]
    assert "wolf_prob" in system and "Strategy" in system


def test_decisions_are_logged(tmp_path, monkeypatch):
    import agent.agent as A
    monkeypatch.setattr(A, "LOG_DIR", tmp_path)
    llm = FakeLLM({"action_id": "0", "reasoning_summary": "x"})
    MyAgent(llm, log=lambda *_: None).choose_action(state("Your role: VILLAGER", legal=["0", "4", "7"]), CTX)
    rows = [l for f in tmp_path.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert len(rows) == 1 and '"choice": "0"' in rows[0]


def test_lessons_go_into_prompt(tmp_path, monkeypatch):
    import agent.agent as A
    (tmp_path / "lessons.md").write_text("- Don't trust a day-1 seer claim blindly.\n")
    monkeypatch.setattr(A, "LESSONS_FILE", tmp_path / "lessons.md")
    monkeypatch.setattr(A, "LOG_DIR", tmp_path)
    llm = FakeLLM({"action_id": "4", "reasoning_summary": "x"})
    MyAgent(llm, log=lambda *_: None).choose_action(state("Your role: VILLAGER", legal=["0", "4", "7"]), CTX)
    assert "day-1 seer claim" in llm.calls[0][0]["content"]


def test_dry_run(monkeypatch):
    from agent.agent import DryRunProvider
    agent = MyAgent(DryRunProvider(), log=lambda *_: None)
    out = agent.choose_action(state("Your role: VILLAGER", legal=["0", "4", "7"]), CTX)
    assert out.action.action_id == "0"


def test_wolf_sees_seer_suspects():
    b = Notebook()
    b.update(state("Your role: WOLF\nYour fellow wolves: Player5"),
             [{"from": "Player3", "text": "I am the seer, 5 is wolf"}])
    assert b.summary()["seer_suspects"] == [3]


def test_opponents_remembered_across_games(tmp_path):
    dead = [{"day": 1, "player": "Bot1", "role": "wolf", "cause": "vote"}]
    b = Notebook()
    b.update(state("Your role: VILLAGER", dead=dead, votes=[{"day": 1, "votes": {"Bot3": "Bot1"}}],
                   alive=[0, 2, 3, 4, 5, 6]), [{"from": "Player6", "text": "I'm the seer"}])
    opps = Opponents(tmp_path / "o.json")
    opps.save_game("g1", b.this_game())
    opps.save_game("g1", b.this_game())      # 同一場存兩次不會重複算
    st = Opponents(tmp_path / "o.json").stats(skip_session="g2")
    assert st["Bot3"]["votes_hit_wolf"] == 1 and st["Bot1"]["wolf_games"] == 1
    assert st["Bot6"]["games"] == 1 and st["Bot6"]["fake_seer"] == 0  # 角色沒揭露就不算假跳
    assert Opponents(tmp_path / "o.json").stats(skip_session="g1") == {}


def test_real_name_format():
    # 平台真實的名字長這樣 (第一場 dry run 的 log)
    name = lambda i: f"Simon Liu (Player{i}, Self-hosted)"
    s = state("Your role: VILLAGER", alive=[0, 2, 3],
              dead=[{"day": 1, "player": name(1), "role": "wolf", "cause": "vote"}])
    s.raw["players"] = [{"position": i, "name": name(i)} for i in range(7)]
    s.raw["game_state"]["alive"] = [name(i) for i in (0, 2, 3)]
    b = Notebook()
    b.update(s, [])
    assert b.alive == {0, 2, 3} and b.dead[1][0] == "wolf"
    assert b.names[4] == "Simon Liu"  # 跨場用的名字不含座位號
