"use strict";

// This worker receives a player observation and public events. It has no live Battle.
const {performance} = require("node:perf_hooks");
const {Battle, Dex, Teams} = require("pokemon-showdown");
const {playStep} = require("./planning_rollout.cjs");
const {evaluate: evaluatePreview} = require("./planning_preview.cjs");
const {value: estimatePosition} = require("./planning_value.cjs");

const SEATS = ["p1", "p2"];
const ROTATION_CHANGES = new Set(["replace", "swap", "detailschange", "-formechange", "-transform", "-terastallize"]);
const SIMPLE_STATUS = new Set(["", "brn", "par", "psn"]);
const WEATHER_ROCKS = {raindance: "damprock", sunnyday: "heatrock", sandstorm: "smoothrock", snowscape: "icyrock"};
const SIMPLE_TERRAINS = new Set(["grassyterrain", "psychicterrain", "electricterrain", "mistyterrain"]);
const SCREENS = new Set(["reflect", "lightscreen"]);
const RESTORED_VOLATILES = new Set(["protosynthesis", "quarkdrive", "glaiverush", "taunt", "encore", "yawn", "leechseed"]);
const id = value => String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
const locationOf = actor => /^(p[12])([ab])?: (.+)$/.exec(actor || "");

function trackField(timers, parts, tick, nicknames, extended) {
    const [, event, actor, detail, ...args] = parts;
    function source(tags) {
        const owner = tags.find(tag => tag?.startsWith("[of] "))?.slice(5);
        const mover = locationOf(owner);
        return mover && nicknames.get(`${mover[1]}:${mover[3]}`);
    }
    if (event === "-weather") {
        if ([detail, ...args].includes("[upkeep]")) return;
        timers.delete("weather");
        const condition = id(actor);
        if (condition === "none") return;
        const record = source([detail, ...args]);
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
        else {
            const record = source(args) || [...nicknames.values()].findLast(mon => mon.lastMove === condition);
            const duration = condition === "tailwind" ? 4 : extended && SCREENS.has(condition)
                ? (record?.item === "lightclay" ? 8 : 5) : null;
            timers.set(key, {condition, kind: "side", seat, expires: duration === null ? null : tick + duration});
        }
    } else if (event === "-fieldstart" || event === "-fieldend") {
        const condition = id(actor.replace(/^move: /i, ""));
        if (event === "-fieldend") timers.delete(condition);
        else {
            const terrain = extended && SIMPLE_TERRAINS.has(condition);
            const duration = condition === "trickroom" ? 5 : terrain ? (source([detail, ...args])?.item === "terrainextender" ? 8 : 5) : null;
            timers.set(condition, {condition, kind: terrain ? "terrain" : "field",
                                   expires: duration === null ? null : tick + duration});
        }
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

function readPublic(events, cards, extended) {
    const sides = Object.fromEntries(SEATS.map(seat => [seat, {team: new Map(), active: [null, null]}]));
    const nicknames = new Map();
    const timers = new Map();
    let turn = 0;
    let tick = 0;
    let lastMoveActor;
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
            trackField(timers, parts, tick, nicknames, extended);
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
                record = {species, item: id(cards.get(species).item), moves: new Map(), sleepTicks: 0, poisonTicks: 0};
                side.team.set(species, record);
                nicknames.set(key, record);
            }
            record.entered = turn;
            record.actions = 0;
            record.poisonTicks = 0;
            record.lastMove = "";
            record.lastProtect = -2;
            record.protects = 0;
            record.effects = new Map();
            record.fallen = 0;
            side.active[slot === "a" ? 0 : 1] = record;
            if (!health(record, args[0])) return {reason: "unknown_hp"};
        } else if (record) {
            if (event === "-damage" || event === "-heal" || event === "-sethp") {
                if (!health(record, detail)) return {reason: "unknown_hp"};
                if (args.includes("[from] psn") || args.includes("[from] tox")) record.poisonTicks++;
            } else if (event === "faint") {
                record.health = [0, 0];
                record.status = "";
                record.effects.clear();
            } else if (event === "move") {
                const move = id(detail);
                record.moves.set(move, (record.moves.get(move) || 0) + 1);
                record.lastMove = move;
                record.actions++;
                lastMoveActor = location;
            } else if (event === "cant") {
                record.actions++;
                if (detail === "slp") record.sleepTicks++;
            } else if (event === "-item" || event === "-enditem") {
                record.item = event === "-item" ? id(detail) : "";
            } else if (event === "-status" || event === "-curestatus") {
                record.status = event === "-status" ? id(detail) : "";
                if (event === "-status") {
                    record.sleepTicks = 0;
                    record.poisonTicks = 0;
                }
            } else if (event === "-singleturn" && id(detail.replace(/^move: /i, "")) === "protect") {
                record.protects = record.lastProtect === turn - 1 ? record.protects + 1 : 1;
                record.lastProtect = turn;
            } else if (event === "-start" || event === "-end" || event === "-singlemove") {
                const effect = id(detail.replace(/^(?:move|ability): /i, ""));
                const normalized = effect.replace(/^(protosynthesis|quarkdrive)(atk|def|spa|spd|spe)$/, "$1");
                if (/^fallen\d$/.test(effect)) record.fallen = event === "-end" ? 0 : Number(effect.at(-1));
                else if (event === "-end") record.effects.delete(normalized);
                else if (event !== "-singlemove" || !extended) {
                    const stat = normalized === effect ? undefined : effect.slice(normalized.length);
                    const sourceSlot = lastMoveActor?.[1] && lastMoveActor?.[2] ? lastMoveActor[1] + lastMoveActor[2] : undefined;
                    record.effects.set(normalized, {tick, stat, sourceSlot, move: record.lastMove,
                        fromBooster: args.includes("[fromitem]") || args.includes("[from] item: Booster Energy") || !record.item});
                }
            } else if (event === "-ability" && id(detail) !== id(cards.get(record.species).ability)) {
                return {reason: "unsupported_event"};
            }
        }
    }
    return {sides, turn, tick, timers: [...timers.values()].map(timer => ({...timer,
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
    const extended = Boolean(message.lineups);
    const publicState = readPublic(message.events, cards, extended);
    if (publicState.reason) return publicState;
    if (publicState.turn !== board.turn) return {reason: "stale_history"};
    if (publicState.timers.some(timer => timer.duration === null)) return {reason: "unsupported_field"};
    const activeTimers = (kind, seat) => publicState.timers.filter(timer => timer.kind === kind && timer.seat === seat)
        .map(timer => timer.condition).sort().join(",");
    const other = message.seat === "p1" ? "p2" : "p1";
    if (publicState.timers.some(timer => !(timer.duration > 0)) || activeTimers("weather") !== board.weather.join(",") ||
        [activeTimers("field"), activeTimers("terrain")].filter(Boolean).sort().join(",") !== board.fields.join(",") ||
        activeTimers("side", message.seat) !== board.side_conditions.join(",") ||
        activeTimers("side", other) !== board.opponent_side_conditions.join(",")) return {reason: "field_mismatch"};
    const ourIndex = SEATS.indexOf(message.seat);
    const rosters = [];
    for (const [index, seat] of SEATS.entries()) {
        const {team, active} = publicState.sides[seat];
        if ((!extended && team.size !== 4) || active.some(record => !record)) return {reason: "unrevealed_roster"};
        if ([...team.keys()].some(species => !message.rosters[index].includes(species))) return {reason: "roster_mismatch"};
        const records = [...active, ...[...team.values()].filter(record => !active.includes(record))];
        if (extended) {
            const lineup = message.lineups[index];
            if (new Set(lineup).size !== 4 || lineup.some(species => !message.rosters[index].includes(species)) ||
                records.some(record => !lineup.includes(record.species))) return {reason: "roster_mismatch"};
            for (const species of lineup) {
                if (team.has(species)) continue;
                records.push({species, item: id(cards.get(species).item), health: [100, 100], status: "",
                              moves: new Map(), effects: new Map(), unseen: true});
            }
        }
        if (new Set(records).size !== 4) return {reason: "slot_mismatch"};
        const observed = index === ourIndex ? board.active : board.opponents;
        for (const [slot, record] of active.entries()) {
            const mon = observed[slot];
            if (record.health[0] ? mon?.species !== record.species : mon && !mon.fainted) return {reason: "slot_mismatch"};
            if (record.health[0] && (mon.status || "") !== record.status) return {reason: "status_mismatch"};
            if (index === ourIndex && record.health[0] && (mon.item || "") !== record.item) return {reason: "item_mismatch"};
            const ability = id(cards.get(record.species).ability);
            if (record.health[0] && ((!extended && record.effects.size) ||
                [...record.effects.keys()].some(effect => !RESTORED_VOLATILES.has(effect)) ||
                (!extended && ability === "unburden" && !record.item))) {
                return {reason: "active_effect"};
            }
            record.boosts = mon?.boosts;
        }
        if (records.some(record => !SIMPLE_STATUS.has(record.status) &&
            !(extended && ["frz", "slp", "tox"].includes(record.status)))) return {reason: "status_counter"};
        rosters.push(records);
    }

    const root = new Battle({formatid: message.format, seed: [1, 2, 3, 4], send: () => {}});
    try {
        for (const [index, seat] of SEATS.entries()) {
            root.setPlayer(seat, {name: seat, team: Teams.pack(rosters[index].map(record => cards.get(record.species)))});
        }
        const proxy = message.proxy_seat ? root.getSide(message.proxy_seat) : null;
        const previewRequest = proxy?.activeRequest;
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
                const range = record.unseen ? [pokemon.maxhp, pokemon.maxhp] : hpRange(pokemon, record, ours ? ourFraction : null);
                if (!range) return {reason: "hp_mismatch"};
                hp.push(range);
                pokemon.hp = Math.floor((range[0] + range[1]) / 2);
                pokemon.fainted = !pokemon.hp;
                pokemon.status = record.status;
                pokemon.statusState = root.initEffectState({id: record.status, target: pokemon});
                if (extended && record.status === "slp") pokemon.statusState.time = Math.max(1, 3 - (record.sleepTicks || 0));
                if (extended && record.status === "tox") pokemon.statusState.stage = Math.min(15, record.poisonTicks || 0);
                pokemon.item = record.item;
                pokemon.itemState = root.initEffectState({id: record.item, target: pokemon});
                pokemon.abilityState = root.initEffectState({id: pokemon.ability, target: pokemon, fallen: record.fallen});
                pokemon.volatiles = {};
                for (const key of Object.keys(pokemon.boosts)) pokemon.boosts[key] = record.boosts?.[key] || 0;
                pokemon.activeTurns = slot < 2 ? Math.max(1, root.turn - record.entered) : 0;
                pokemon.activeMoveActions = slot < 2 ? Math.max(record.actions, pokemon.activeTurns - 1) : 0;
                pokemon.lastMove = slot < 2 && record.lastMove ? root.dex.getActiveMove(record.lastMove) : null;
                if (slot < 2 && pokemon.getItem().isChoice && pokemon.lastMove && record.lastMove !== "struggle") {
                    pokemon.volatiles.choicelock = root.initEffectState({id: "choicelock", target: pokemon, move: record.lastMove});
                }
                if (slot < 2 && record.lastProtect === root.turn - 1) {
                    pokemon.volatiles.stall = root.initEffectState({id: "stall", target: pokemon, duration: 1,
                        counter: 3 ** Math.min(record.protects, 6)});
                }
                if (extended && slot < 2 && !pokemon.fainted) {
                    const tick = publicState.tick;
                    for (const [effect, state] of record.effects) {
                        const value = {id: effect, target: pokemon};
                        if (state.stat) Object.assign(value, {bestStat: state.stat, fromBooster: state.fromBooster});
                        if (effect === "taunt") value.duration = Math.max(1, 3 + state.tick - tick);
                        if (effect === "encore") Object.assign(value, {duration: Math.max(1, 3 + state.tick - tick), move: state.move});
                        if (effect === "yawn") value.duration = Math.max(1, 2 + state.tick - tick);
                        if (effect === "glaiverush") value.duration = 1;
                        if (effect === "leechseed") value.sourceSlot = state.sourceSlot;
                        pokemon.volatiles[effect] = root.initEffectState(value);
                    }
                    if (pokemon.ability === "unburden" && !pokemon.item) {
                        pokemon.volatiles.unburden = root.initEffectState({id: "unburden", target: pokemon});
                    }
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
            } else if (timer.kind === "terrain") {
                root.field.terrain = timer.condition;
                root.field.terrainState = effect;
            } else {
                target.sideConditions[timer.condition] = effect;
            }
        }
        for (const mon of root.getAllActive()) {
            if (!mon.fainted) root.runEvent("DisableMove", mon);
        }
        root.makeRequest("move");
        const views = root.sides.map(view);
        if (proxy) views[SEATS.indexOf(proxy.id)].pivot = {request: proxy.activeRequest, preview_request: previewRequest};
        return {views, state: JSON.stringify(root.toJSON()), ranges, fields: fieldSummary(root)};
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

function positionValue(battle, seat) {
    if (battle.ended) return battle.winner ? (battle.winner === seat ? 5 : -5) : 0;
    const our = battle.getSide(seat);
    const foe = our.foe;
    const room = battle.field.pseudoWeather.trickroom?.duration || 0;
    function sideValue(side) {
        let score = 0;
        for (const mon of side.pokemon) {
            if (mon.fainted) continue;
            const fraction = mon.hp / mon.maxhp;
            const physical = mon.baseMoveSlots.filter(slot => battle.dex.moves.get(slot.id).category === "Physical").length;
            const status = {slp: .2, frz: .34, tox: .21, par: .14, brn: physical ? .24 : .10, psn: .14}[mon.status] || 0;
            score += .48 + .87 * fraction - status * Math.sqrt(fraction) - .12 * Boolean(mon.volatiles.leechseed);
            if (!mon.isActive) continue;
            const {atk, spa, def, spd, spe} = mon.boosts;
            const offense = physical > 1 ? atk : physical === 1 ? (atk + spa) / 2 : spa;
            score += Math.sqrt(fraction) * (.11 * offense + .045 * (def + spd) + .055 * spe * (room ? -1 : 1));
            score += .075 * Boolean(mon.volatiles.quarkdrive || mon.volatiles.protosynthesis);
            score += .10 * (mon.activeMoveActions === 0 && mon.moves.includes("fakeout"));
            score -= .15 * Boolean(mon.volatiles.yawn);
        }
        score += .044 * ((side.sideConditions.reflect?.duration || 0) + (side.sideConditions.lightscreen?.duration || 0));
        return score;
    }
    const speeds = side => side.active.filter(mon => mon && !mon.fainted).map(mon => mon.getActionSpeed());
    let tempo = 0;
    for (const own of speeds(our)) {
        for (const theirs of speeds(foe)) tempo += Math.sign(own - theirs) * (room ? -1 : 1);
    }
    return sideValue(our) - sideValue(foe) + .065 * tempo;
}

function branch(snapshot, choices, seed, hpSample, seat, choiceText, movingOptions, audit, planning, interaction, payoff) {
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
            if (planning >= 2) {
                if (!playStep(battle, movingOptions, choiceText)) return null;
            } else {
                battle.makeChoices();
            }
        }
        if (!battle.ended && (battle.turn <= turn || battle.requestState !== "move")) return null;
        if (audit) return {turn: battle.turn, fields: fieldSummary(battle)};
        if (!planning) return leaf(battle, seat);
        const immediate = payoff(battle, seat);
        if (battle.ended || planning < 2) return immediate;
        if (!playStep(battle, movingOptions, choiceText, interaction)) return immediate;
        for (let substep = 0; !battle.ended && battle.requestState === "switch" && substep < 4; substep++) {
            if (!playStep(battle, movingOptions, choiceText)) break;
        }
        return .38 * immediate + .62 * payoff(battle, seat);
    } finally {
        battle.destroy();
    }
}

module.exports = ({view, choiceText, movingOptions}) => {
    let cards;
    let snapshot;
    let model;
    const worlds = new Map();
    return message => {
        if (message.command === "configure") {
            cards = new Map(message.cards.map(card => [id(card.species), card]));
            model = message.position_model;
            if (cards.size !== message.cards.length) throw new Error("Public search requires one published set per species");
            return {ready: true};
        }
        if (message.command === "plan_preview") {
            return evaluatePreview(message.options, message.opponents, message.seeds, message.format, cards,
                                   choiceText, movingOptions, positionValue, message.budget_ms);
        }
        if (message.command === "prepare") {
            snapshot = null;
            if (message.world !== undefined) worlds.delete(message.world);
            try {
                snapshot = prepare(message, cards, view);
            } catch (error) {
                return {reason: "prepare_error", detail: error.message};
            }
            if (snapshot.reason) return {reason: snapshot.reason};
            if (message.world !== undefined) worlds.set(message.world, snapshot);
            return {views: snapshot.views, fields: snapshot.fields};
        }
        if (message.command !== "evaluate") throw new Error("Unknown public search command");
        const world = message.world === undefined ? snapshot : worlds.get(message.world);
        if (!world?.state) throw new Error("No prepared public search state");
        const deadline = performance.now() + message.budget_ms;
        const foe = SEATS.find(seat => seat !== message.seat);
        const matrix = [];
        const payoff = !model ? positionValue : (battle, seat) => {
            const base = positionValue(battle, seat);
            return battle.ended ? base : .68 * estimatePosition(battle, seat, model) + .32 * base;
        };
        for (const ours of message.ours) {
            const row = [];
            for (const theirs of message.replies) {
                const samples = [];
                for (const [index, seed] of message.seeds.entries()) {
                    if (performance.now() >= deadline) return {reason: "budget", branches: 0};
                    let value;
                    try {
                        const hpSample = message.spread_hp ? (message.seeds.length === 1 ? .5 : index / (message.seeds.length - 1)) : index / 2;
                        value = branch(world, {[message.seat]: ours, [foe]: theirs}, seed, hpSample, message.seat,
                            choiceText, movingOptions, message.audit_fields, message.planning, message.interaction, payoff);
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
