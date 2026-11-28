"use strict";

const readline = require("node:readline");
const {performance} = require("node:perf_hooks");
const {Battle, Teams} = require("pokemon-showdown");
const {choiceText, movingOptions} = require("./showdown.cjs");
const {playStep} = require("./planning_rollout.cjs");
const {value} = require("./planning_value.cjs");

function prepare(option, cards) {
    const order = option.leads.concat(option.bring.filter(name => !option.leads.includes(name)));
    return Teams.pack(order.map(name => {
        const card = cards.get(name);
        if (!card) throw new Error(`Unknown public card ${name}`);
        return card;
    }));
}

function rollout(own, foe, seed, format, depth, interaction, model) {
    const battle = new Battle({formatid: format, seed});
    try {
        battle.setPlayer("p1", {name: "p1", team: own});
        battle.setPlayer("p2", {name: "p2", team: foe});
        battle.makeChoices("team 1234", "team 1234");
        for (let step = 0; !battle.ended && battle.turn <= depth && step < 4 * depth; step++) {
            if (!playStep(battle, movingOptions, choiceText, interaction)) {
                throw new Error("Preview rollout produced an illegal choice");
            }
        }
        if (battle.ended) return {value: battle.winner ? Number(battle.winner === "p1") : .5, terminal: true};
        return {value: .5 + value(battle, "p1", model) / 10, terminal: false};
    } finally {
        battle.destroy();
    }
}

function mean(values) {
    return values.reduce((total, current) => total + current, 0) / values.length;
}

function score(values, weights) {
    const averages = values.map(mean);
    const total = weights.reduce((a, b) => a + b, 0);
    const expected = averages.reduce((sum, average, i) => sum + average * weights[i] / total, 0);
    return .88 * expected + .12 * Math.min(...averages);
}

function measure(message, cards, model) {
    const own = message.options.map(option => prepare(option, cards));
    const foes = message.opponents.map(option => prepare(option, cards));
    const deadline = performance.now() + message.budget_ms;
    let indices = own.map((_, i) => i);
    let completed = [];
    let calls = 0, terminals = 0, timedOut = false;
    for (const {survivors, depth, start, samples, opponentCount} of message.stages) {
        const next = [];
        const selected = foes.slice(0, opponentCount);
        for (const index of indices) {
            const values = [];
            for (const foe of selected) {
                const draws = [];
                for (let sample = start; sample < start + samples; sample++) {
                    if (performance.now() >= deadline) {
                        timedOut = true;
                        break;
                    }
                    const result = rollout(own[index], foe, message.seeds[sample], message.format, depth, sample % 3 !== 2, model);
                    calls++;
                    terminals += Number(result.terminal);
                    draws.push(result.value);
                }
                if (timedOut) break;
                values.push(draws);
            }
            if (timedOut) break;
            next.push({index, values, score: score(values, message.weights.slice(0, selected.length))});
        }
        if (timedOut) break;
        next.sort((a, b) => b.score - a.score);
        completed = next;
        indices = next.slice(0, survivors).map(row => row.index);
        if (!indices.includes(0)) indices[indices.length - 1] = 0;
    }
    const anchor = completed.find(row => row.index === 0);
    const best = completed[0];
    const improvement = best && anchor ? best.score - anchor.score : 0;
    return {calls, terminals, timedOut, rows: completed, improvement, index: improvement >= message.edge ? best.index : 0};
}

let cards, model;
readline.createInterface({input: process.stdin}).on("line", line => {
    try {
        const query = JSON.parse(line);
        let response;
        if (query.command === "configure") {
            cards = new Map(query.cards.map(card => [card.species.toLowerCase().replace(/[^a-z0-9]/g, ""), card]));
            model = query.position_model;
            response = {ready: true};
        } else if (query.command === "compare") {
            response = measure(query, cards, model);
        } else {
            throw new Error("Unknown Preview research command");
        }
        process.stdout.write(JSON.stringify(response) + "\n");
    } catch (error) {
        process.stdout.write(JSON.stringify({error: error.message}) + "\n");
    }
});
