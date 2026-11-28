"use strict";

const DOUBLES_MOVES = ["fakeout", "ragepowder", "followme", "spore", "encore", "tailwind", "trickroom"];
const STATUS = ["brn", "par", "slp", "frz", "psn", "tox"];

function sum(group, fn) {
    return group.reduce((total, item) => total + fn(item), 0);
}

function hp(mon) { return mon.hp / mon.maxhp; }

function duration(condition) { return Math.min(5, condition?.duration ?? 0); }

function statusPressure(mon) {
    const steps = mon.status === "slp" ? .45 : mon.status === "par" ? .75 : 1;
    const offense = mon.status === "brn" ? .5 : 1;
    return {steps, offense};
}

function stage(value) {
    return (2 + Math.max(0, value)) / (2 - Math.min(0, value));
}

function estimate(attacker, victim, pressure) {
    const boosts = attacker.boosts;
    const condition = statusPressure(attacker);
    const physical = stage(boosts.atk) * condition.offense;
    const special = stage(boosts.spa);
    const physicalShare = attacker.baseMoveSlots.filter(slot => attacker.battle.dex.moves.get(slot.id).category === "Physical").length;
    const specialShare = attacker.baseMoveSlots.filter(slot => attacker.battle.dex.moves.get(slot.id).category === "Special").length;
    const enemy = victim.boosts;
    const screens = victim.side.sideConditions;
    const physicalFactor = physical / stage(enemy.def) * (screens.reflect ? .67 : 1);
    const specialFactor = special / stage(enemy.spd) * (screens.lightscreen ? .67 : 1);
    const scale = (physicalShare * physicalFactor + specialShare * specialFactor) / Math.max(1, physicalShare + specialShare);
    const row = pressure[attacker.species.id] ?? pressure[attacker.baseSpecies.id];
    const raw = row?.[victim.species.id] ?? row?.[victim.baseSpecies.id] ?? 0;
    return Math.min(2, condition.steps * scale * raw);
}

function oneSide(side, pressure) {
    const alive = mon => mon && !mon.fainted && mon.hp > 0;
    const front = side.active.filter(alive);
    const enemyFront = side.foe.active.filter(alive);
    const back = side.pokemon.filter(mon => alive(mon) && !front.includes(mon));
    const team = [...front, ...back];
    const remaining = side.foe.pokemon.filter(alive);
    const room = side.battle.field.pseudoWeather.trickroom;
    const order = room ? -1 : 1;
    const speeds = front.map(mon => mon.getActionSpeed());
    const theirSpeed = enemyFront.map(mon => mon.getActionSpeed());
    const faster = sum(speeds, speed => theirSpeed.filter(other => order * (speed - other) > 0).length);
    const ratios = sum(speeds, speed => sum(theirSpeed, other => order * Math.tanh(Math.log(Math.max(1, speed) / Math.max(1, other)))));
    const outgoing = enemyFront.map(mon => front.map(actor => estimate(actor, mon, pressure)));
    const incoming = front.map(mon => enemyFront.map(actor => estimate(actor, mon, pressure)));
    const pressureScores = outgoing.map((values, i) => Math.min(hp(enemyFront[i]), Math.max(0, ...values)));
    const koScores = outgoing.map((values, i) => Math.min(1.3, sum(values, value => value) / hp(enemyFront[i])));
    const safeTargets = incoming.map((values, i) => Math.min(1.2, Math.max(0, ...values) / hp(front[i])));
    const reserveCounter = enemyFront.map(victim => Math.max(0, ...back.map(mon => estimate(mon, victim, pressure))));
    const coverage = remaining.map(victim => Math.max(0, ...team.map(mon => estimate(mon, victim, pressure))));
    const abilityTempo = ["intimidate", "regenerator", "drought", "drizzle", "sandstream", "snowwarning"];
    return [team.length, front.length, sum(team, hp), sum(team, mon => Math.sqrt(hp(mon))), sum(front, hp), sum(back, hp),
        faster, ratios, sum(pressureScores, v => v), sum(koScores, v => v), sum(safeTargets, v => v),
        sum(reserveCounter, v => v), sum(coverage, v => v), Math.min(0.9, Math.max(0, ...coverage)),
        ...STATUS.map(status => sum(team, mon => mon.status === status ? hp(mon) : 0)),
        ...["atk", "spa", "def", "spd", "spe"].map(stat => sum(front, mon => mon.boosts[stat] * hp(mon))),
        ...["tailwind", "reflect", "lightscreen", "auroraveil"].map(key => duration(side.sideConditions[key])),
        ...DOUBLES_MOVES.map(move => sum(front, mon => mon.moves.includes(move) ? Math.sqrt(hp(mon)) : 0)),
        sum(front, mon => mon.moves.includes("fakeout") && !mon.activeMoveActions ? hp(mon) : 0),
        sum(front, mon => mon.volatiles.yawn ? hp(mon) : 0),
        sum(front, mon => mon.volatiles.leechseed ? hp(mon) : 0),
        sum(front, mon => mon.volatiles.taunt ? hp(mon) : 0),
        sum(front, mon => mon.volatiles.quarkdrive || mon.volatiles.protosynthesis ? hp(mon) : 0),
        sum(back, mon => abilityTempo.includes(mon.ability) ? hp(mon) : 0),
        side.sideConditions.spikes?.layers ?? 0, side.sideConditions.stealthrock ? 1 : 0];
}

function features(battle, seat, pressure) {
    const side = battle.getSide(seat);
    const room = duration(battle.field.pseudoWeather.trickroom);
    return [oneSide(side, pressure).concat(room), oneSide(side.foe, pressure).concat(room)];
}

function linear(values, layer) {
    return layer.weight.map((weights, i) => weights.reduce((total, weight, j) => total + weight * values[j], layer.bias[i]));
}

function probability(battle, seat, model) {
    const [own, foe] = features(battle, seat, model.pressure);
    const directional = (a, b) => {
        const x = a.map((value, i) => (value - b[i]) / model.scale[i]);
        const neutral = a.map((value, i) => (value + b[i]) / model.scale[i]);
        let layer = x.concat(neutral);
        for (let i = 0; i < model.layers.length; i++) {
            layer = linear(layer, model.layers[i]);
            if (i + 1 < model.layers.length) layer = layer.map(value => value / (1 + Math.exp(-value)));
        }
        return layer[0];
    };
    return 5 * Math.tanh((directional(own, foe) - directional(foe, own)) / 2);
}

module.exports = {features, probability};
