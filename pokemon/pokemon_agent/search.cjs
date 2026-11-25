"use strict";

// This worker receives a player observation and public events. It has no live Battle.
const {performance} = require("node:perf_hooks");
const {Battle, Dex, Teams} = require("pokemon-showdown");

const SEATS = ["p1", "p2"];
const ROTATION_CHANGES = new Set(["replace", "swap", "detailschange", "-formechange", "-transform", "-terastallize"]);
const SIMPLE_STATUS = new Set(["", "brn", "par", "psn"]);
const WEATHER_ROCKS = {raindance: "damprock", sunnyday: "heatrock", sandstorm: "smoothrock", snowscape: "icyrock"};
const id = value => String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
const locationOf = actor => /^(p[12])([ab])?: (.+)$/.exec(actor || "");

function trackField(timers, parts, tick, nicknames) {
    const [, event, actor, detail, ...args] = parts;
    if (event === "-weather") {
        if ([detail, ...args].includes("[upkeep]")) return;
        timers.delete("weather");
        const condition = id(actor);
        if (condition === "none") return;
        const source = [detail, ...args].find(tag => tag?.startsWith("[of] "))?.slice(5);
        const mover = locationOf(source);
        const record = mover && nicknames.get(`${mover[1]}:${mover[3]}`);
        const rock = WEATHER_ROCKS[condition];
        // In this fixed pool the weather setters are switch-in abilities. Require their public source.
        const duration = rock && record ? Dex.conditions.get(condition).duration + (record.item === rock ? 3 : 0) : null;
        timers.set("weather", {condition, kind: "weather", expires: duration === null ? null : tick + duration});
        return;
    }
    if (event === "-sidestart" || event === "-sideend") {
        const condition = id(detail.replace(/^move: /i, ""));
        const seat = actor.split(":")[0];
        const key = `${seat}:${condition}`;
        if (event === "-sideend") timers.delete(key);
        else timers.set(key, {condition, kind: "side", seat,
                             expires: condition === "tailwind" ? tick + Dex.moves.get(condition).condition.duration : null});
    } else if (event === "-fieldstart" || event === "-fieldend") {
        const condition = id(actor.replace(/^move: /i, ""));
        if (event === "-fieldend") timers.delete(condition);
        else timers.set(condition, {condition, kind: "field",
                                    expires: condition === "trickroom" ? tick + Dex.moves.get(condition).condition.duration : null});
    }
}

function fieldSummary(battle) {
    return {
        weather: battle.field.weather ? {[battle.field.weather]: battle.field.weatherState.duration} : {},
        fields: Object.fromEntries(Object.entries(battle.field.pseudoWeather).map(([key, value]) => [key, value.duration])),
        sides: battle.sides.map(side => Object.fromEntries(Object.entries(side.sideConditions)
            .map(([key, value]) => [key, value.duration]))),
    };
}

function health(record, condition) {
    const fields = /^(\d+)(?:\/(\d+))?(?: (\w+))?$/.exec(condition || "");
    if (!fields) return false;
    const hp = Number(fields[1]);
    const scale = Number(fields[2]);
    if (hp && (!scale || hp > scale)) return false;
    record.health = [hp, scale];
    record.status = hp ? (fields[3] || "") : "";
    return true;
}

function readPublic(events, cards) {
    const sides = Object.fromEntries(SEATS.map(seat => [seat, {team: new Map(), active: [null, null]}]));
    const nicknames = new Map();
    const timers = new Map();
    let turn = 0;
    let tick = 0;
    for (const line of events) {
        const parts = line.split("|");
        const [, event, actor, detail, ...args] = parts;
        if (event === "turn") {
            turn = Number(actor);
            continue;
        }
        if (event === "upkeep") {
            tick++;
            continue;
        }
        if (event === "-weather" || event.startsWith("-field") || event.startsWith("-side")) {
            trackField(timers, parts, tick, nicknames);
            continue;
        }
        if (ROTATION_CHANGES.has(event) || event === "-cureteam" || event === "-endability") return {reason: "unsupported_event"};
        const location = locationOf(actor);
        if (!location) continue;
        const [, seat, slot, name] = location;
        const key = `${seat}:${name}`;
        const side = sides[seat];
        let record = nicknames.get(key);
        if (event === "switch" || event === "drag") {
            const species = id(detail.split(",")[0]);
            if (!slot || !cards.has(species) || (record && record.species !== species)) return {reason: "unknown_card"};
            if (!record) {
                record = {species, item: id(cards.get(species).item), moves: new Map()};
                side.team.set(species, record);
                nicknames.set(key, record);
            }
            record.entered = turn;
            record.actions = 0;
            record.lastMove = "";
            record.lastProtect = -2;
            record.protects = 0;
            record.effects = new Set();
            record.fallen = 0;
            side.active[slot === "a" ? 0 : 1] = record;
            if (!health(record, args[0])) return {reason: "unknown_hp"};
        } else if (record) {
            if (event === "-damage" || event === "-heal" || event === "-sethp") {
                if (!health(record, detail)) return {reason: "unknown_hp"};
            } else if (event === "faint") {
                record.health = [0, 0];
                record.status = "";
                record.effects.clear();
            } else if (event === "move") {
                const move = id(detail);
                record.moves.set(move, (record.moves.get(move) || 0) + 1);
                record.lastMove = move;
                record.actions++;
            } else if (event === "cant") {
                record.actions++;
            } else if (event === "-item" || event === "-enditem") {
                record.item = event === "-item" ? id(detail) : "";
            } else if (event === "-status" || event === "-curestatus") {
                record.status = event === "-status" ? id(detail) : "";
            } else if (event === "-singleturn" && id(detail.replace(/^move: /i, "")) === "protect") {
                record.protects = record.lastProtect === turn - 1 ? record.protects + 1 : 1;
                record.lastProtect = turn;
            } else if (event === "-start" || event === "-end" || event === "-singlemove") {
                const effect = id(detail.replace(/^(?:move|ability): /i, ""));
                if (/^fallen\d$/.test(effect)) record.fallen = event === "-end" ? 0 : Number(effect.at(-1));
                else if (event === "-end") record.effects.delete(effect);
                else record.effects.add(effect);
            } else if (event === "-ability" && id(detail) !== id(cards.get(record.species).ability)) {
                return {reason: "unsupported_event"};
            }
        }
    }
    return {sides, turn, timers: [...timers.values()].map(timer => ({...timer,
        duration: timer.expires === null ? null : timer.expires - tick}))};
}

function hpRange(pokemon, record, ownExactFraction) {
    if (!record.health[0]) return [0, 0];
    const [shown, scale] = record.health;
    const low = Math.floor((shown - 1) * pokemon.maxhp / scale) + 1;
    const high = Math.floor(shown * pokemon.maxhp / scale);
    if (ownExactFraction == null) return low <= high ? [low, high] : null;
    const hp = Math.round(ownExactFraction * pokemon.maxhp);
    return hp >= low && hp <= high ? [hp, hp] : null;
}

function prepare(message, cards, view) {
    const board = message.observation;
    const publicState = readPublic(message.events, cards);
    if (publicState.reason) return publicState;
    if (publicState.turn !== board.turn) return {reason: "stale_history"};
    if (publicState.timers.some(timer => timer.duration === null)) return {reason: "unsupported_field"};
    const activeTimers = (kind, seat) => publicState.timers.filter(timer => timer.kind === kind && timer.seat === seat)
        .map(timer => timer.condition).sort().join(",");
    const other = message.seat === "p1" ? "p2" : "p1";
    if (publicState.timers.some(timer => !(timer.duration > 0)) || activeTimers("weather") !== board.weather.join(",") ||
        activeTimers("field") !== board.fields.join(",") || activeTimers("side", message.seat) !== board.side_conditions.join(",") ||
        activeTimers("side", other) !== board.opponent_side_conditions.join(",")) return {reason: "field_mismatch"};
    const ourIndex = SEATS.indexOf(message.seat);
    const rosters = [];
    for (const [index, seat] of SEATS.entries()) {
        const {team, active} = publicState.sides[seat];
        if (team.size !== 4 || active.some(record => !record)) return {reason: "unrevealed_roster"};
        if ([...team.keys()].some(species => !message.rosters[index].includes(species))) return {reason: "roster_mismatch"};
        const records = [...active, ...[...team.values()].filter(record => !active.includes(record))];
        if (new Set(records).size !== 4) return {reason: "slot_mismatch"};
        const observed = index === ourIndex ? board.active : board.opponents;
        for (const [slot, record] of active.entries()) {
            const mon = observed[slot];
            if (record.health[0] ? mon?.species !== record.species : mon && !mon.fainted) return {reason: "slot_mismatch"};
            if (record.health[0] && (mon.status || "") !== record.status) return {reason: "status_mismatch"};
            if (index === ourIndex && record.health[0] && (mon.item || "") !== record.item) return {reason: "item_mismatch"};
            const ability = id(cards.get(record.species).ability);
            if (record.health[0] && (record.effects.size || (ability === "unburden" && !record.item))) {
                return {reason: "active_effect"};
            }
            record.boosts = mon?.boosts;
        }
        if (records.some(record => !SIMPLE_STATUS.has(record.status))) return {reason: "status_counter"};
        rosters.push(records);
    }

    const root = new Battle({formatid: message.format, seed: [1, 2, 3, 4], send: () => {}});
    try {
        for (const [index, seat] of SEATS.entries()) {
            root.setPlayer(seat, {name: seat, team: Teams.pack(rosters[index].map(record => cards.get(record.species)))});
        }
        root.makeChoices("team 1234", "team 1234");
        root.turn = board.turn;
        root.field.weather = "";
        root.field.weatherState = root.initEffectState({id: ""});
        root.field.terrain = "";
        root.field.terrainState = root.initEffectState({id: ""});
        root.field.pseudoWeather = {};
        const ranges = [];
        for (const [index, side] of root.sides.entries()) {
            const ours = index === ourIndex;
            const hp = [];
            for (const [slot, pokemon] of side.pokemon.entries()) {
                const record = rosters[index][slot];
                const ourFraction = ours && (slot < 2 ? board.active[slot]?.hp : board.reserves?.[record.species]?.hp);
                const range = hpRange(pokemon, record, ours ? ourFraction : null);
                if (!range) return {reason: "hp_mismatch"};
                hp.push(range);
                pokemon.hp = Math.floor((range[0] + range[1]) / 2);
                pokemon.fainted = !pokemon.hp;
                pokemon.status = record.status;
                pokemon.statusState = root.initEffectState({id: record.status, target: pokemon});
                pokemon.item = record.item;
                pokemon.itemState = root.initEffectState({id: record.item, target: pokemon});
                pokemon.abilityState = root.initEffectState({id: pokemon.ability, target: pokemon, fallen: record.fallen});
                pokemon.volatiles = {};
                for (const key of Object.keys(pokemon.boosts)) pokemon.boosts[key] = record.boosts?.[key] || 0;
                pokemon.activeTurns = slot < 2 ? root.turn - record.entered : 0;
                pokemon.activeMoveActions = slot < 2 ? Math.max(record.actions, pokemon.activeTurns - 1) : 0;
                pokemon.lastMove = slot < 2 && record.lastMove ? root.dex.getActiveMove(record.lastMove) : null;
                if (slot < 2 && pokemon.getItem().isChoice && pokemon.lastMove && record.lastMove !== "struggle") {
                    pokemon.volatiles.choicelock = root.initEffectState({id: "choicelock", target: pokemon, move: record.lastMove});
                }
                if (slot < 2 && record.lastProtect === root.turn - 1) {
                    pokemon.volatiles.stall = root.initEffectState({id: "stall", target: pokemon, duration: 1,
                        counter: 3 ** Math.min(record.protects, 6)});
                }
                for (const move of [...pokemon.moveSlots, ...pokemon.baseMoveSlots]) {
                    const knownPP = ours ? message.own_pp[slot]?.[move.id] : undefined;
                    move.pp = Number.isInteger(knownPP) ? knownPP : Math.max(0, move.maxpp - (record.moves.get(move.id) || 0));
                    move.disabled = false;
                    move.disabledSource = "";
                }
            }
            side.totalFainted = side.pokemon.filter(mon => mon.fainted).length;
            side.pokemonLeft = side.pokemon.length - side.totalFainted;
            ranges.push(hp);
        }
        for (const timer of publicState.timers) {
            const target = timer.kind === "side" ? root.getSide(timer.seat) : root.field;
            const effect = root.initEffectState({id: timer.condition, target, duration: timer.duration});
            if (timer.kind === "weather") {
                root.field.weather = timer.condition;
                root.field.weatherState = effect;
            } else if (timer.kind === "field") {
                root.field.pseudoWeather[timer.condition] = effect;
            } else {
                target.sideConditions[timer.condition] = effect;
            }
        }
        for (const mon of root.getAllActive()) {
            if (!mon.fainted) root.runEvent("DisableMove", mon);
        }
        root.makeRequest("move");
        return {views: root.sides.map(view), state: JSON.stringify(root.toJSON()), ranges, fields: fieldSummary(root)};
    } finally {
        root.destroy();
    }
}

function leaf(battle, seat) {
    if (battle.ended) return battle.winner ? (battle.winner === seat ? 8 : -8) : 0;
    const our = battle.getSide(seat);
    const foe = our.foe;
    const material = side => side.pokemon.reduce((value, mon) => value + (mon.fainted ? 0 : .55 + mon.hp / mon.maxhp), 0);
    let tempo = 0;
    for (const ally of our.active.filter(mon => mon && !mon.fainted)) {
        for (const enemy of foe.active.filter(mon => mon && !mon.fainted)) {
            tempo += Math.sign(ally.getActionSpeed() - enemy.getActionSpeed());
        }
    }
    return material(our) - material(foe) + .04 * tempo;
}

function branch(snapshot, choices, seed, hpSample, seat, choiceText, audit) {
    const battle = Battle.fromJSON(snapshot.state);
    battle.restart(() => {});
    battle.resetRNG(seed);
    try {
        for (const [i, side] of battle.sides.entries()) {
            for (const [j, pokemon] of side.pokemon.entries()) {
                const [low, high] = snapshot.ranges[i][j];
                pokemon.hp = Math.round(low + hpSample * (high - low));
            }
        }
        const turn = battle.turn;
        const commands = battle.sides.map(side => choiceText(side, choices[side.id]));
        for (const [index, side] of battle.sides.entries()) {
            if (!battle.choose(side.id, commands[index])) return null;
        }
        for (let substep = 0; !battle.ended && battle.requestState === "switch" && substep < 4; substep++) {
            battle.makeChoices();
        }
        if (!battle.ended && (battle.turn <= turn || battle.requestState !== "move")) return null;
        return audit ? {turn: battle.turn, fields: fieldSummary(battle)} : leaf(battle, seat);
    } finally {
        battle.destroy();
    }
}

module.exports = ({view, choiceText}) => {
    let cards;
    let snapshot;
    return message => {
        if (message.command === "configure") {
            cards = new Map(message.cards.map(card => [id(card.species), card]));
            if (cards.size !== message.cards.length) throw new Error("Public search requires one published set per species");
            return {ready: true};
        }
        if (message.command === "prepare") {
            snapshot = null;
            try {
                snapshot = prepare(message, cards, view);
            } catch (error) {
                return {reason: "prepare_error", detail: error.message};
            }
            return snapshot.reason ? {reason: snapshot.reason} : {views: snapshot.views, fields: snapshot.fields};
        }
        if (message.command !== "evaluate" || !snapshot?.state) throw new Error("No prepared public search state");
        const deadline = performance.now() + message.budget_ms;
        const foe = SEATS.find(seat => seat !== message.seat);
        const matrix = [];
        for (const ours of message.ours) {
            const row = [];
            for (const theirs of message.replies) {
                const samples = [];
                for (const [index, seed] of message.seeds.entries()) {
                    if (performance.now() >= deadline) return {reason: "budget", branches: 0};
                    let value;
                    try {
                        value = branch(snapshot, {[message.seat]: ours, [foe]: theirs}, seed, index / 2, message.seat,
                            choiceText, message.audit_fields);
                    } catch (error) {
                        return {reason: "branch_error", detail: error.message};
                    }
                    if (message.audit_fields ? value === null : !Number.isFinite(value)) return {reason: "branch_illegal"};
                    samples.push(value);
                }
                row.push(samples);
            }
            matrix.push(row);
        }
        if (performance.now() >= deadline) return {reason: "budget", branches: 0};
        return {matrix};
    };
};
