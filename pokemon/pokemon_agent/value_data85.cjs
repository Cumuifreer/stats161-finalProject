"use strict";

const readline = require("node:readline");
const {Battle, Teams} = require("pokemon-showdown");
const {choiceText, movingOptions} = require("./showdown.cjs");
const {playStep} = require("./planning_rollout.cjs");
const {features} = require("./position85.cjs");

function simulate(spec, pressure, format) {
    const battle = new Battle({formatid: format, seed: spec.seed});
    try {
        for (let i = 0; i < 2; i++) {
            const seat = `p${i + 1}`;
            battle.setPlayer(seat, {name: seat, team: Teams.pack(spec.teams[i])});
        }
        battle.makeChoices("team 1234", "team 1234");
        let last = 0;
        const states = [];
        for (let step = 0; !battle.ended && step < 120 && battle.turn <= 32; step++) {
            if (battle.requestState === "move" && battle.turn > last) {
                states.push(features(battle, "p1", pressure));
                last = battle.turn;
            }
            const strategic = spec.style === 0 || (spec.style === 2 && (battle.turn + spec.seed[0]) % 4 !== 0);
            if (!playStep(battle, movingOptions, choiceText, strategic)) return {winner: null, states: []};
        }
        return {winner: battle.ended ? (battle.winner || "draw") : null, states};
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
            const samples = query.games.map(spec => simulate(spec, pressure, query.format));
            response = {samples, cpu: process.cpuUsage(started)};
        } else {
            throw new Error("Unknown synthetic position data command");
        }
        process.stdout.write(JSON.stringify(response) + "\n");
    } catch (error) {
        process.stdout.write(JSON.stringify({error: error.message}) + "\n");
    }
});
