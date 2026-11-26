"use strict";

const STATUSES = ["brn", "par", "slp", "frz", "psn", "tox"];
const CONDITIONS = ["tailwind", "reflect", "lightscreen", "spikes"];
const BOOSTS = ["atk", "spa", "def", "spd", "spe"];
const MOVE_GROUPS = [
    new Set(["tailwind", "trickroom"]),
    new Set(["fakeout", "spore", "encore"]),
    new Set(["recover", "protect", "calmmind", "nastyplot"]),
];

function effectiveSpeed(mon, tailwind) {
    const stage = mon.boosts.spe;
    const speed = Math.floor(mon.storedStats.spe * (2 + Math.max(0, stage)) / (2 + Math.max(0, -stage)));
    return speed * (tailwind ? 2 : 1) * (mon.status === "par" ? .5 : 1) * (mon.item === "choicescarf" ? 1.5 : 1);
}

function sideFeatures(side, order, pressure) {
    const alive = mon => mon && !mon.fainted && mon.hp > 0;
    const active = side.active.filter(alive);
    const reserves = side.pokemon.slice(2).filter(alive);
    const living = active.concat(reserves);
    const enemies = side.foe.active.filter(alive);
    const allEnemies = side.foe.pokemon.filter(alive);
    const hp = mon => mon.hp / mon.maxhp;
    const rootHp = mon => Math.sqrt(hp(mon));
    const sum = (mons, compute) => mons.reduce((total, mon) => total + compute(mon), 0);
    const theirSpeeds = enemies.map(mon => effectiveSpeed(mon, side.foe.sideConditions.tailwind));
    const faster = sum(active, mon => theirSpeeds.filter(speed =>
        order * (effectiveSpeed(mon, side.sideConditions.tailwind) - speed) > 0).length);
    const offense = mon => Math.max(0, ...enemies.map(enemy => {
        const damage = pressure[mon.species.id][enemy.species.id];
        return Math.min(damage, hp(enemy)) + .20 * (damage >= hp(enemy));
    }));
    const coverage = sum(allEnemies, enemy => Math.max(0, ...living.map(mon => pressure[mon.species.id][enemy.species.id])));
    const states = STATUSES.map(status => sum(living, mon => mon.status === status ? rootHp(mon) : 0));
    const boosted = BOOSTS.map(stat => (stat === "spe" ? order : 1) *
        sum(active, mon => mon.boosts[stat] * rootHp(mon)));
    const conditions = CONDITIONS.map(condition => Number(Boolean(side.sideConditions[condition])));
    const sets = MOVE_GROUPS.map(group => sum(active, mon => mon.baseMoveSlots.some(move => group.has(move.id)) ? rootHp(mon) : 0));
    return [living.length, sum(living, hp), sum(living, rootHp), sum(active, hp), faster,
        sum(active, offense), sum(reserves, offense), coverage, ...states, ...boosted, ...conditions, ...sets];
}

function value(battle, seat, model) {
    const player = battle.getSide(seat);
    const order = battle.field.pseudoWeather.trickroom ? -1 : 1;
    const lift = sideFeatures(player, order, model.pressure);
    const drop = sideFeatures(player.foe, order, model.pressure);
    const logit = model.weights.reduce((total, weight, i) => total + weight * (lift[i] - drop[i]), 0);
    return 5 * Math.tanh(logit / 2);
}

module.exports = {value};
