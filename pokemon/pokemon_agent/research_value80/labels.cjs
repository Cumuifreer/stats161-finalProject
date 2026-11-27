"use strict";

const readline = require("node:readline");
const {Battle, Teams} = require("pokemon-showdown");
const {choiceText, movingOptions} = require("../showdown.cjs");
const {movement, playStep} = require("../planning_rollout.cjs");
const {features} = require("../position85.cjs");
const {policyChoice, joinChoice} = require(process.argv[2]);

function initialize(seed) {
    let state = seed >>> 0;
    return () => {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        return (state >>> 0) / 4294967296;
    };
}

function sample(battle, side, strategy, rng) {
    if (side.activeRequest?.wait || !side.activeRequest) return "";
    const template = movingOptions(side, side.activeRequest);
    if (strategy === "monte-carlo" && battle.requestState === "move") {
        const plan = policyChoice(battle, side.n, rng, 17);
        const legal = plan.every((part, i) => template[i].options.some(option => {
            if (part.kind !== option.type) return false;
            if (part.kind === "move") return part.id === option.move_id && option.targets.includes(part.target);
            if (part.kind === "switch") return part.species === option.species;
            return true;
        }));
        if (legal) return joinChoice(plan);
    }
    const selected = movement(side, template, strategy !== "damage");
    return choiceText(side, Array.isArray(selected) ? selected[0]?.action : selected);
}

function simulate(spec, pressure, format) {
    const battle = new Battle({formatid: format, seed: spec.seed});
    const rng = initialize((spec.seed[0] << 16) | spec.seed[1] | 1);
    const lineup = [["joint", "monte-carlo"], ["monte-carlo", "joint"], ["monte-carlo", "damage"]][spec.style];
    try {
        for (let i = 0; i < 2; i++) {
            const seat = `p${i + 1}`;
            battle.setPlayer(seat, {name: seat, team: Teams.pack(spec.teams[i])});
        }
        battle.makeChoices("team 1234", "team 1234");
        let last = 0;
        const states = [];
        for (let step = 0; !battle.ended && step < 145 && battle.turn <= 45; step++) {
            if (battle.requestState === "move" && battle.turn > last) {
                states.push(features(battle, "p1", pressure));
                last = battle.turn;
            }
            if (battle.requestState === "switch") {
                if (!playStep(battle, movingOptions, choiceText, true)) return {winner: null, states: []};
            } else {
                const options = battle.sides.map((side, i) => sample(battle, side, lineup[i], rng));
                battle.makeChoices(...options);
            }
        }
        return {winner: battle.ended ? (battle.winner || "draw") : null, states};
    } catch {
        return {winner: null, states: []};
    } finally {
        battle.destroy();
    }
}

let pressure;
readline.createInterface({input: process.stdin}).on("line", line => {
    try {
        const query = JSON.parse(line);
        let response;
        if (query.command === "configure") {
            pressure = query.pressure;
            response = {ready: true};
        } else if (query.command === "label") {
            const started = process.cpuUsage();
            response = {samples: query.games.map(game => simulate(game, pressure, query.format)), cpu: process.cpuUsage(started)};
        } else {
            throw new Error("Unknown synthetic position data command");
        }
        process.stdout.write(JSON.stringify(response) + "\n");
    } catch (error) {
        process.stdout.write(JSON.stringify({error: error.message}) + "\n");
    }
});
