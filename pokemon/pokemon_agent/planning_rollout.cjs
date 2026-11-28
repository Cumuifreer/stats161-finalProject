"use strict";

const {choose: chooseInteractions} = require("./planning_interaction.cjs");

// A cheap continuation policy for hypothetical worlds. Root transitions still use Showdown.
const PROTECT = new Set(["protect", "detect"]);
const PIVOT = new Set(["uturn", "partingshot"]);
const SPREAD = new Set(["allAdjacent", "allAdjacentFoes"]);
const PRIORITY_GUARD = new Set(["armortail", "dazzling", "queenlymajesty"]);

function matchup(battle, attacker, defender) {
    if (!attacker || !defender) return 0;
    return Math.max(...attacker.getTypes().map(kind => {
        if (!defender.runImmunity(kind)) return 0;
        return 2 ** defender.getTypes().reduce((stages, type) =>
            stages + battle.dex.getEffectiveness(kind, type), 0);
    }));
}

function switchScore(side, candidate, replacement) {
    const foes = side.foe.active.filter(mon => mon && !mon.fainted);
    const matchupScore = foes.map(mon => Math.log2(Math.max(.25, matchup(side.battle, candidate, mon))) -
        Math.log2(Math.max(.25, matchup(side.battle, mon, candidate))));
    const cycle = Number(candidate.ability === "intimidate") * .14 + Number(candidate.moves.includes("fakeout")) * .1;
    return (replacement ? .3 : .09) * (candidate.hp / candidate.maxhp) +
        .13 * matchupScore.reduce((a, b) => a + b, 0) + cycle;
}

function replacements(side, slots) {
    const available = side.pokemon.slice(2).filter(mon => !mon.fainted);
    const sorted = available.sort((a, b) => switchScore(side, b, true) - switchScore(side, a, true));
    const action = {};
    for (const {slot, options} of slots) {
        const best = sorted.findIndex(mon => options.some(option => option.type === "switch" && option.species === mon.species.id));
        action["slot_" + slot] = best < 0 ? {type: "pass"} : {type: "switch", species: sorted.splice(best, 1)[0].species.id};
    }
    return action;
}

function estimateDamage(battle, attacker, defender, raw, spread) {
    if (!defender || defender.fainted) return 0;
    const move = battle.dex.getActiveMove(raw);
    battle.singleEvent("ModifyType", move, null, attacker, defender, move, move);
    battle.runEvent("ModifyType", attacker, defender, move, move);
    if (!move.basePower && !move.damageCallback) return 0;
    move.spreadHit = spread;
    if (move.id !== "surgingstrikes") move.willCrit = false;
    else move.willCrit = true;
    const damage = battle.actions.getDamage(attacker, defender, move, true) || 0;
    const hits = Array.isArray(move.multihit) ? (attacker.item === "loadeddice" ? 4.5 : 3) : move.multihit || 1;
    return hits * damage * (move.accuracy === true ? 1 : move.accuracy / 100);
}

function blockedPriority(battle, side, source, target, move) {
    if (!target || target.side === side) return false;
    const mods = (source.ability === "prankster" && move.category === "Status")
        || (source.ability === "galewings" && source.hp === source.maxhp && move.type === "Flying")
        || (source.ability === "triage" && move.flags.heal);
    if (move.priority <= 0 && !mods) return false;
    return (battle.field.terrain === "psychicterrain" && target.isGrounded()) ||
        side.foe.active.some(mon => mon && !mon.fainted && PRIORITY_GUARD.has(mon.ability)) ||
        (source.ability === "prankster" && move.category === "Status" && target.hasType("Dark"));
}

function movement(side, slots, interaction) {
    const battle = side.battle;
    const roster = [side.active, side.foe.active];
    const alive = mon => mon && !mon.fainted;
    const speeds = roster.map(team => team.map(mon => alive(mon) ? mon.getActionSpeed() : 0));
    const room = battle.field.pseudoWeather.trickroom ? -1 : 1;
    const originalRng = battle.prng;
    battle.prng = originalRng.clone();
    const options = [];
    try {
        for (const {slot, options: moves} of slots) {
            const attacker = side.active[slot];
            const partner = side.active[1 - slot];
            const faster = speeds[1].filter(speed => speed > 0 && room * (speeds[0][1 - slot] - speed) > 0).length;
            const pressures = side.foe.active.filter(alive).map(mon => matchup(battle, mon, attacker));
            const pressure = Math.max(.5, ...pressures);
            const ranked = [];
            for (const {type, species, move_id, targets} of moves) {
                if (type === "pass") {
                    ranked.push({atom: {type}, value: 0, hits: [0, 0]});
                } else if (type === "switch") {
                    const incoming = side.pokemon.find(mon => mon.species.id === species);
                    const improvement = switchScore(side, incoming, false) - switchScore(side, attacker, false);
                    const hp = attacker.hp / attacker.maxhp;
                    ranked.push({atom: {type, species}, value: improvement + .07 * pressure * (1 - hp) - .09, hits: [0, 0]});
                } else {
                    const move = battle.dex.moves.get(move_id);
                    for (const target of targets.length ? targets : [0]) {
                        const foe = target > 0 ? roster[1][target - 1] : null;
                        const other = target < 0 ? roster[0][-target - 1] : null;
                        const atom = {type, move_id, target};
                        const hits = [0, 0];
                        let value = 0;
                        if (blockedPriority(battle, side, attacker, foe, move)) {
                            ranked.push({atom, value: -.75, hits});
                            continue;
                        }
                        if (move.category !== "Status") {
                            const spread = SPREAD.has(move.target);
                            const multi = spread && (roster[1].filter(alive).length + Number(move.target === "allAdjacent" && alive(partner)) > 1);
                            if (move_id === "fakeout" && attacker.activeMoveActions > 0) value -= .8;
                            else for (let f = 0; f < 2; f++) {
                                const enemy = roster[1][f];
                                if (!alive(enemy) || (!spread && target !== f + 1 && target !== 0)) continue;
                                const damage = estimateDamage(battle, attacker, enemy, move_id, multi);
                                hits[f] = Math.min(1.1, damage / enemy.maxhp);
                                if (damage && (move.priority > 0 || room * (speeds[0][slot] - speeds[1][f]) > 0)) {
                                    value += .15 * Math.min(1, damage / enemy.hp);
                                }
                                if (move_id === "fakeout" && damage && !["innerfocus", "shielddust"].includes(enemy.ability) &&
                                    enemy.item !== "covertcloak") value += .36;
                            }
                            if (move.target === "allAdjacent" && alive(partner)) {
                                const damage = estimateDamage(battle, attacker, partner, move_id, multi);
                                value -= .85 * damage / partner.maxhp + .3 * (damage >= partner.hp);
                            } else if (other) value -= .6;
                            if (move_id === "suckerpunch") {
                                hits[0] *= .76;
                                hits[1] *= .76;
                            }
                            if (move_id === "icywind" && faster < 2 && room > 0) value += .12;
                            if (PIVOT.has(move_id) && side.pokemonLeft > side.active.filter(alive).length) value += .10;
                        } else if (PROTECT.has(move_id)) {
                            value = attacker.volatiles.stall ? -.2 : .07 * pressure / Math.max(.4, attacker.hp / attacker.maxhp);
                        } else if (move_id === "tailwind") {
                            value = !side.sideConditions.tailwind && room > 0 ? .18 * (2 - faster) : -.4;
                        } else if (move_id === "trickroom") {
                            value = room > 0 ? .18 * (2 - 2 * faster) : .08 * (1 - faster);
                        } else if (["reflect", "lightscreen"].includes(move_id)) {
                            value = side.sideConditions[move_id] ? -.3 : .25;
                        } else if (["spore", "yawn", "willowisp", "toxic", "thunderwave", "leechseed"].includes(move_id)) {
                            value = alive(foe) && !foe.status &&
                                !(move_id === "spore" && (foe.hasType("Grass") || foe.item === "safetygoggles"))
                                ? (move_id === "spore" ? .43 : .26) * (.4 + .6 * foe.hp / foe.maxhp) : -.25;
                        } else if (move_id === "recover") {
                            value = Math.min(.5, 1 - attacker.hp / attacker.maxhp) - .08 * pressure;
                        } else if (["nastyplot", "calmmind"].includes(move_id)) {
                            value = .38 - .095 * pressure / Math.max(.2, attacker.hp / attacker.maxhp);
                        } else if (move_id === "helpinghand") {
                            value = alive(partner) ? .16 : -.3;
                        } else if (["taunt", "encore", "ragepowder", "followme", "partingshot"].includes(move_id)) {
                            value = .08;
                        }
                        ranked.push({atom, value, hits});
                    }
                }
            }
            options.push(ranked);
        }
    } finally {
        battle.prng = originalRng;
    }
    let bestScore = -Infinity;
    let best;
    const ranked = [];
    for (const left of options[0]) {
        for (const right of options[1]) {
            if (left.atom.species && left.atom.species === right.atom.species) continue;
            let score = left.value + right.value;
            for (let f = 0; f < 2; f++) {
                const enemy = side.foe.active[f];
                if (!alive(enemy)) continue;
                const damage = left.hits[f] + right.hits[f];
                const hp = enemy.hp / enemy.maxhp;
                score += Math.min(hp, damage) + .34 * Math.min(1, damage / hp) ** 3;
            }
            const shared = left.atom.move_id === right.atom.move_id;
            if (shared && PROTECT.has(left.atom.move_id)) score -= .23;
            if (shared && ["tailwind", "trickroom", "reflect", "lightscreen"].includes(left.atom.move_id)) score -= .5;
            const action = {slot_0: left.atom, slot_1: right.atom};
            if (interaction) ranked.push({action, score, entries: [left, right]});
            if (score > bestScore) {
                bestScore = score;
                best = action;
            }
        }
    }
    return interaction ? ranked.sort((a, b) => b.score - a.score) : best;
}

function playStep(battle, optionSlots, choiceText, interaction = false) {
    const jointMove = interaction && battle.sides.every(side => side.activeRequest && !side.activeRequest.wait &&
        !side.activeRequest.forceSwitch);
    const options = battle.sides.map(side => {
        const request = side.activeRequest;
        if (!request || request.wait) return null;
        const slots = optionSlots(side, request);
        return request.forceSwitch ? replacements(side, slots) : movement(side, slots, jointMove);
    });
    const actions = jointMove ? chooseInteractions(battle, options, estimateDamage) : options;
    const commands = actions.map((action, index) => action ? choiceText(battle.sides[index], action) : null);
    for (const [index, side] of battle.sides.entries()) {
        if (commands[index] && !battle.choose(side.id, commands[index])) return false;
    }
    return commands.some(Boolean);
}

module.exports = {playStep, movement};
