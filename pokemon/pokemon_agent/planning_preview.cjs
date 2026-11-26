"use strict";

const {performance} = require("node:perf_hooks");
const {Battle, Teams} = require("pokemon-showdown");
const {playStep} = require("./planning_rollout.cjs");

function evaluate(options, opponents, seeds, format, cards, choiceText, movingOptions, positionValue, budgetMs) {
    const deadline = performance.now() + budgetMs;
    const own = options.map(option => pack(option, cards));
    const foes = opponents.map(option => pack(option, cards));
    const scores = own.map((_, index) => ({index, draws: []}));
    const phases = [
        {survivors: Math.min(12, own.length), scenarios: [0, 1], draws: 1, depth: 2},
        {survivors: Math.min(4, own.length), scenarios: foes.map((_, i) => i), draws: 2, depth: 4},
        {survivors: 1, scenarios: foes.map((_, i) => i), draws: seeds.length - 3, depth: 6},
    ];
    let seedIndex = 0, calls = 0, current = scores;
    for (const {survivors, scenarios, draws, depth} of phases) {
        if (!draws || current.length === 1) break;
        const stage = [];
        const opponentsToTest = scenarios.filter(index => index < foes.length);
        for (const {index} of current) {
            const margins = [];
            for (const opponentIndex of opponentsToTest) {
                const group = [];
                for (let draw = 0; draw < draws; draw++) {
                    if (performance.now() >= deadline) return answer(current, calls, true);
                    group.push(play(own[index], foes[opponentIndex], seeds[seedIndex + draw], format, depth,
                                    choiceText, movingOptions, positionValue));
                    calls++;
                }
                margins.push(group);
            }
            stage.push({index, margins, score: score(margins)});
        }
        seedIndex += draws;
        stage.sort((a, b) => b.score - a.score);
        current = stage;
        if (survivors > 1 && stage.length > survivors) {
            const best = stage.slice(0, survivors);
            if (!best.some(row => row.index === 0)) best[best.length - 1] = stage.find(row => row.index === 0);
            current = best;
        }
    }
    return answer(current, calls, false);
}

function score(rows) {
    const margins = rows.map(draws => draws.reduce((sum, value) => sum + value, 0) / draws.length);
    const expected = margins.reduce((sum, value) => sum + value, 0) / margins.length;
    return .83 * expected + .17 * Math.min(...margins);
}

function answer(ranked, calls, timedOut) {
    const anchor = ranked.find(row => row.index === 0);
    const best = ranked[0];
    const advantage = best.score !== undefined && anchor.score !== undefined ? best.score - anchor.score : 0;
    return {index: advantage > .13 ? best.index : 0, calls, advantage, timedOut};
}

function pack(action, cards) {
    const ordered = action.leads.concat(action.bring.filter(mon => !action.leads.includes(mon)));
    return Teams.pack(ordered.map(species => {
        if (!cards.has(species)) throw new Error("Lineup species absent from the public catalog");
        return cards.get(species);
    }));
}

function play(own, foe, seed, format, depth, choiceText, movingOptions, positionValue) {
    const battle = new Battle({formatid: format, seed, send: () => {}});
    try {
        battle.setPlayer("p1", {name: "p1", team: own});
        battle.setPlayer("p2", {name: "p2", team: foe});
        battle.makeChoices("team 1234", "team 1234");
        for (let step = 0; !battle.ended && battle.turn <= depth && step < 4 * depth; step++) {
            if (!playStep(battle, movingOptions, choiceText)) {
                throw new Error("Preview continuation produced an illegal choice");
            }
        }
        return Math.tanh(positionValue(battle, "p1") / 2);
    } finally {
        battle.destroy();
    }
}

module.exports = {evaluate};
