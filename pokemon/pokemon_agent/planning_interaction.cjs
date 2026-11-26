"use strict";

const DEFENSE = new Set(["protect", "detect"]);
const HELP = new Set(["ragepowder", "followme", "helpinghand", "spore", "willowisp", "taunt"]);

function candidates(ranked, size = 7) {
    const options = ranked.slice(0, 3);
    for (const fits of [
        atom => DEFENSE.has(atom.move_id),
        atom => atom.type === "switch",
        atom => atom.move_id === "fakeout" || HELP.has(atom.move_id),
    ]) {
        const row = ranked.find(candidate => candidate.entries.some(entry => fits(entry.atom)));
        if (row && !options.includes(row)) options.push(row);
    }
    return options.concat(ranked.filter(row => !options.includes(row))).slice(0, size);
}

function choose(battle, ranked, damage) {
    const menus = ranked.map(rows => candidates(rows));
    const active = battle.sides.flatMap(side => side.active);
    const order = battle.field.pseudoWeather.trickroom ? -1 : 1;
    const speeds = active.map(mon => mon && !mon.fainted ? mon.getActionSpeed() : 0);
    const cache = new Map();
    const prior = battle.prng;
    battle.prng = prior.clone();
    let table;
    try {
        table = menus[0].map(left => menus[1].map(right =>
            margin(battle, active, speeds, order, left.entries.concat(right.entries), damage, cache)));
    } finally {
        battle.prng = prior;
    }
    const columns = menus[1].map((_, col) => table.map(row => -row[col]));
    const aggregates = [table, columns].map(rows => rows.map(values => {
        const average = (values[0] * 2 + values.slice(1).reduce((sum, value) => sum + value, 0)) / (values.length + 1);
        return .82 * average + .18 * Math.min(...values);
    }));
    return menus.map((menu, side) => menu[aggregates[side].indexOf(Math.max(...aggregates[side]))].action);
}

function margin(battle, active, speeds, order, choices, damage, cache) {
    const mons = active.slice();
    const occupied = mon => mon && !mon.fainted;
    let score = 0;
    const pending = [];
    for (const [position, entry] of choices.entries()) {
        if (!occupied(mons[position])) continue;
        const side = position < 2 ? 0 : 1;
        const sign = side ? -1 : 1;
        if (entry.atom.type === "switch") {
            mons[position] = battle.sides[side].pokemon.find(mon => mon.species.id === entry.atom.species);
            score += sign * entry.value;
        } else if (entry.atom.type === "move") {
            const move = battle.dex.moves.get(entry.atom.move_id);
            const priority = move.priority + Number(mons[position].ability === "prankster" && move.category === "Status");
            pending.push({position, priority, move});
        }
    }
    pending.sort((a, b) => b.priority - a.priority || order * (speeds[b.position] - speeds[a.position]));
    const hp = mons.map(mon => occupied(mon) ? mon.hp / mon.maxhp : 0);
    const stopped = new Set(), protectedSlots = new Set();
    for (const {position, move} of pending) {
        if (!hp[position] || stopped.has(position)) continue;
        const side = position < 2 ? 0 : 1;
        const sign = side ? -1 : 1;
        const entry = choices[position];
        const mon = mons[position];
        if (DEFENSE.has(move.id)) {
            if (!mon.volatiles.stall) protectedSlots.add(position);
            continue;
        }
        if (move.category === "Status") {
            score += sign * .90 * entry.value;
            continue;
        }
        if (move.id === "fakeout" && mon.activeMoveActions > 0) continue;
        const target = entry.atom.target;
        if (target < 0) score -= .35 * sign;
        for (let slot = 0; slot < 2; slot++) {
            const foePosition = (1 - side) * 2 + slot;
            const foe = mons[foePosition];
            if (hp[foePosition] <= 0 || (target > 0 && target !== slot + 1) || target < 0) continue;
            if (protectedSlots.has(foePosition) && !(mon.ability === "unseenfist" && move.flags.contact)) continue;
            let amount = entry.hits[slot];
            if (foe !== active[foePosition]) {
                const key = [position, move.id, foePosition, foe.species.id].join("/");
                if (!cache.has(key)) {
                    const spread = ["allAdjacent", "allAdjacentFoes"].includes(move.target);
                    cache.set(key, damage(battle, mon, foe, move.id, spread) / foe.maxhp);
                }
                amount = cache.get(key);
            }
            if (!amount) continue;
            const old = hp[foePosition];
            hp[foePosition] = Math.max(0, old - amount);
            score += sign * ((old - hp[foePosition]) + .40 * Number(hp[foePosition] === 0));
            if (move.id === "fakeout" && !["innerfocus", "shielddust"].includes(foe.ability) && foe.item !== "covertcloak") {
                stopped.add(foePosition);
            }
        }
        if (["uturn", "icywind", "knockoff", "snarl"].includes(move.id)) score += .045 * sign;
    }
    return score;
}

module.exports = {choose};
