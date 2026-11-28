"use strict";

const {Battle} = require("pokemon-showdown");

function rankOffense(ranked) {
    const chosen = ranked.slice(0, 2);
    const support = ranked.find(row => [row.action.slot_0, row.action.slot_1].some(atom =>
        atom.type === "switch" || ["protect", "detect", "fakeout", "spore", "encore"].includes(atom.move_id)));
    if (support && !chosen.includes(support)) chosen.push(support);
    if (chosen.length < 3 && ranked.length > 2) chosen.push(ranked[2]);
    return chosen.map(row => row.action);
}

function take(battle, actions, choiceText) {
    const commands = battle.sides.map((side, index) => choiceText(side, actions[index]));
    return battle.sides.every((side, index) => battle.choose(side.id, commands[index]));
}

function finishSwitches(battle, movingOptions, choiceText, advance) {
    for (let step = 0; step < 5 && !battle.ended && battle.requestState === "switch"; step++) {
        if (!advance(battle, movingOptions, choiceText)) break;
    }
}

function rankWithExactTurn(battle, movingOptions, choiceText, movement, advance, payoff, seat, deadline) {
    if (battle.requestState !== "move" || !battle.sides.every(side => side.activeRequest && !side.activeRequest.wait)) return null;
    const menus = battle.sides.map(side => rankOffense(movement(side, movingOptions(side, side.activeRequest), true)));
    if (menus.some(options => !options.length)) return null;
    const snapshot = JSON.stringify(battle.toJSON());
    const table = [];
    for (const left of menus[0]) {
        const row = [];
        for (const right of menus[1]) {
            if (performance.now() >= deadline) return null;
            const trial = Battle.fromJSON(snapshot);
            trial.restart(() => {});
            // The estimate uses a separate synthetic stream from the rollout it is steering.
            trial.resetRNG([14821, 7737, battle.turn & 65535, 30097]);
            try {
                if (!take(trial, [left, right], choiceText)) return null;
                finishSwitches(trial, movingOptions, choiceText, advance);
                row.push(payoff(trial, "p1"));
            } finally {
                trial.destroy();
            }
        }
        table.push(row);
    }
    const transpose = menus[1].map((_, index) => table.map(row => -row[index]));
    const scores = [table, transpose].map(rows => rows.map(values => {
        const expected = (2 * values[0] + values.slice(1).reduce((sum, value) => sum + value, 0)) / (values.length + 1);
        return .82 * expected + .18 * Math.min(...values);
    }));
    const indices = scores.map(values => values.indexOf(Math.max(...values)));
    const actions = menus.map((menu, index) => menu[indices[index]]);
    if (!take(battle, actions, choiceText)) return null;
    finishSwitches(battle, movingOptions, choiceText, advance);
    return payoff(battle, seat);
}

function probabilityValue(battle, seat, payoff) {
    if (battle.ended) return payoff(battle, seat);
    const score = payoff(battle, seat);
    return 5 * Math.tanh(score / 2.6);
}

function continuation(battle, movingOptions, choiceText, movement, advance, payoff, seat, deadline) {
    const first = probabilityValue(battle, seat, payoff);
    if (battle.ended || performance.now() >= deadline) return first;
    const endgame = Math.max(...battle.sides.map(side => side.pokemonLeft)) <= 2;
    const value = (position, actor) => probabilityValue(position, actor, payoff);
    const exact = endgame ? rankWithExactTurn(battle, movingOptions, choiceText, movement, advance, value, seat, deadline) : null;
    if (exact === null) {
        if (!advance(battle, movingOptions, choiceText, true)) return first;
        finishSwitches(battle, movingOptions, choiceText, advance);
    }
    const intermediate = value(battle, seat);
    for (let step = 0; step < (endgame ? 3 : 1) && !battle.ended && performance.now() < deadline; step++) {
        if (!advance(battle, movingOptions, choiceText, true)) break;
        finishSwitches(battle, movingOptions, choiceText, advance);
    }
    return (endgame ? .16 : .29) * first + .19 * intermediate + (endgame ? .65 : .52) * value(battle, seat);
}

module.exports = {continuation};
