"use strict";

// A local Showdown battle, driven through newline-delimited JSON over stdio.
// Only each side's pending choice and its player-visible board cross the bridge.
const readline = require("node:readline");
const {Battle, Dex, Teams} = require("pokemon-showdown");
const {extractChannelMessages} = require("pokemon-showdown/dist/sim/battle");

if (process.argv[2] === "--card-stats") {
    const {readFileSync} = require("node:fs");
    const calculator = new Battle({formatid: "gen9vgc2025regi", seed: [0, 0, 0, 1]});
    const stats = JSON.parse(readFileSync(0, "utf8")).map(card => {
        const species = calculator.dex.species.get(card.species);
        const evs = {}, ivs = {};
        for (const stat of Object.keys(species.baseStats)) {
            evs[stat] = calculator.clampIntRange(card.evs?.[stat] ?? 0, 0, 255);
            ivs[stat] = calculator.clampIntRange(card.ivs?.[stat] ?? 31, 0, 31);
        }
        const level = calculator.clampIntRange(card.level || 100, 1, 9999);
        const spread = calculator.spreadModify(species.baseStats, {evs, ivs, level, nature: card.nature});
        if (species.maxHP) spread.hp = species.maxHP;
        return {level, stats: spread};
    });
    process.stdout.write(JSON.stringify(stats) + "\n");
    return;
}

if (process.argv[2] === "--dex") {
    const types = Object.fromEntries(Dex.types.all().map(type => [
        type.id, Object.fromEntries(Object.entries(type.damageTaken).map(([attack, code]) => [
            attack.toLowerCase(), [1, 2, 0.5, 0][code],
        ])),
    ]));
    const species = Object.fromEntries(Dex.species.all().map(mon => [
        mon.id, {types: mon.types.map(type => type.toLowerCase()), ...mon.baseStats},
    ]));
    const moves = Object.fromEntries(Dex.moves.all().map(move => [move.id, {
        type: move.type.toLowerCase(), multihit: move.multihit,
        base_power: move.basePower, accuracy: move.accuracy === true ? 1 : move.accuracy / 100,
        category: move.category.toLowerCase(), priority: move.priority, target: move.target.toLowerCase(),
        contact: Boolean(move.flags.contact),
    }]));
    process.stdout.write(JSON.stringify({types, species, moves}) + "\n");
    return;
}

let battle;
let logPosition;
let pivotSeat;
let pivotPreview;
let searchSeats;
let searchEvents;
let proxySeats;
let proposalPreviews;
let labelRoot;
const labelMode = process.argv[2] === "--counterfactual-labels";
const revealed = new Map();
const previewRosters = new Map();
const publicEvents = new Set([
    "turn", "switch", "drag", "replace", "swap", "move", "cant", "faint", "-damage", "-heal", "-sethp",
    "-status", "-curestatus", "-cureteam", "-item", "-enditem", "-start", "-end", "-singleturn", "-singlemove",
    "detailschange", "-formechange", "-transform", "-terastallize", "-ability", "-endability",
    "upkeep", "-weather", "-fieldstart", "-fieldend", "-sidestart", "-sideend",
]);

function pokemonKey(name) {
    return name.replace(/^(p[12])[ab]: /, "$1: ");
}

function trackReveals() {
    const messages = extractChannelMessages(battle.log.slice(logPosition).join("\n"), [0])[0];
    logPosition = battle.log.length;
    for (const line of messages) {
        const [, event, actor, detail, ...tags] = line.split("|");
        if (searchSeats.length && publicEvents.has(event)) searchEvents.push(line);
        if (event === "poke") {
            if (!previewRosters.has(actor)) previewRosters.set(actor, []);
            const species = detail.split(",")[0].replace(/-\*$/, "");
            previewRosters.get(actor).push(battle.dex.species.get(species));
            continue;
        }
        let field;
        let value = detail;
        let owner = actor;
        if (event === "move") field = "moves";
        if (event === "-ability") field = "ability";
        if (event === "-item") field = "item";
        const item = tags.find(tag => tag.startsWith("[from] item: "));
        if (!field && item) {
            field = "item";
            value = item.slice("[from] item: ".length);
            const source = tags.find(tag => tag.startsWith("[of] "));
            if (source) owner = source.slice("[of] ".length);
        }
        if (!field || !owner || !value) continue;
        const key = pokemonKey(owner);
        if (!revealed.has(key)) revealed.set(key, {moves: new Set()});
        const known = revealed.get(key);
        if (field === "moves") known.moves.add(battle.toID(value));
        else known[field] = battle.toID(value);
    }
}

function publicPokemon(mon, own, previewSpecies) {
    if (!mon || mon.fainted) return null;
    const apparent = own ? mon : (mon.illusion || mon);
    const species = previewSpecies || apparent.species;
    const hp = mon.hp / mon.maxhp;
    const view = {
        species: species.id,
        types: previewSpecies ? species.types : apparent.getTypes(),
        status: mon.status,
        boosts: mon.boosts,
        fainted: mon.fainted,
        current_hp_fraction: own ? hp : previewSpecies ? 0 : Math.ceil(100 * hp) / 100,
    };
    if (own) {
        view.item = mon.item;
        view.ability = mon.ability;
        view.moves = mon.moveSlots.map(move => move.id);
    } else {
        const known = revealed.get(pokemonKey(apparent.fullname));
        const abilities = Object.values(species.abilities);
        view.ability = known?.ability || (abilities.length === 1 ? abilities[0] : undefined);
        view.item = known?.item;
        view.moves = [...(known?.moves || [])];
    }
    return view;
}

function movingOptions(side, request) {
    const battle = side.battle;
    const reserves = side.pokemon.slice(side.active.length).filter(mon => !mon.fainted);
    const switches = reserves.map(mon => ({type: "switch", species: mon.species.id}));
    const force = request.forceSwitch;
    return side.active.map((mon, slot) => {
        const options = [];
        if (force) {
            if (force[slot]) options.push(...switches);
            if (!force[slot] || reserves.length < force.filter(Boolean).length) options.push({type: "pass"});
        } else if (!request.active?.[slot] || mon.fainted || mon.volatiles.commanding) {
            options.push({type: "pass"});
        } else {
            const active = request.active[slot];
            for (const offered of active.moves) {
                if (offered.disabled) continue;
                const move = battle.dex.moves.get(offered.id);
                const target = offered.target || move.target;
                const targets = battle.actions.targetTypeChoices(target)
                    ? [-2, -1, 1, 2].filter(loc => battle.validTargetLoc(loc, mon, target))
                    : [0];
                options.push({
                    type: "move", move_id: offered.id, targets, target,
                    base_power: move.basePower, accuracy: move.accuracy === true ? 1 : move.accuracy / 100,
                    category: move.category, priority: move.priority,
                });
            }
            if (!active.trapped) options.push(...switches);
        }
        return {slot, options};
    });
}

function view(side) {
    const battle = side.battle;
    const request = side.activeRequest;
    if (!request || request.wait || side.isChoiceDone()) return null;
    const opponent = side.foe;
    if (request.teamPreview) {
        const preview = previewRosters.get(opponent.id);
        return {
            seat: side.id,
            phase: "team_preview",
            raw: {
                your_roster: side.pokemon.map(mon => publicPokemon(mon, true)),
                opponent_roster: opponent.pokemon.map((mon, index) => publicPokemon(mon, false, preview[index])),
            },
            templates: [{roster: side.pokemon.map(mon => mon.species.id)}],
        };
    }
    return {
        seat: side.id,
        phase: "moving",
        raw: {
            turn: battle.turn,
            active_pokemon: side.active.map(mon => publicPokemon(mon, true)),
            opponent_active_pokemon: opponent.active.map(mon => publicPokemon(mon, false)),
            team: side.pokemon.slice(side.active.length).map(mon => publicPokemon(mon, true)),
            weather: battle.field.weather,
            fields: [...Object.keys(battle.field.pseudoWeather), battle.field.terrain].filter(Boolean),
            side_conditions: Object.keys(side.sideConditions),
            opponent_side_conditions: Object.keys(opponent.sideConditions),
        },
        templates: [{slots: movingOptions(side, request)}],
    };
}

function choiceText(side, action) {
    const slotOf = species => {
        const index = side.pokemon.findIndex(mon => mon.species.id === species);
        if (index < 0) throw new Error("Species is missing from our Showdown roster");
        return index + 1;
    };
    if (action.type === "select_lineup") {
        const order = action.leads.concat(action.bring.filter(mon => !action.leads.includes(mon)));
        return "team " + order.map(slotOf).join("");
    }
    return [action.slot_0, action.slot_1].map(slot => {
        if (slot.type === "switch") return "switch " + slotOf(slot.species);
        if (slot.type === "pass") return "pass";
        const target = slot.target ? " " + slot.target : "";
        return "move " + slot.move_id + target;
    }).join(", ");
}

function handle(message) {
    if (message.command === "start") {
        labelRoot = undefined;
        battle?.destroy();
        battle = new Battle({formatid: message.format, seed: message.seed});
        revealed.clear();
        previewRosters.clear();
        logPosition = 0;
        if (battle.gameType !== "doubles" || battle.ruleTable.pickedTeamSize !== 4) {
            throw new Error("The selected Showdown format must be doubles bring-4");
        }
        for (const seat of ["p1", "p2"]) {
            battle.setPlayer(seat, {name: seat, team: Teams.pack(message.rosters[seat])});
        }
        pivotSeat = message.pivot_seat;
        pivotPreview = pivotSeat ? battle.getSide(pivotSeat).activeRequest : null;
        searchSeats = message.search_seats || [];
        proxySeats = message.proxy_seats || [];
        proposalPreviews = new Map((message.proposal_seats || []).map(seat => [seat, battle.getSide(seat).activeRequest]));
        searchEvents = [];
    } else if (message.command === "choose") {
        const side = battle.getSide(message.seat);
        if (!battle.choose(side.id, choiceText(side, message.action))) {
            throw new Error(side.choice.error || "Showdown rejected the choice");
        }
    } else if (message.command === "audit_restore" && process.argv[2] === "--audit-restoration") {
        // Keep the complete fixture snapshot inside the simulator process.
        const restored = Battle.fromJSON(JSON.stringify(battle.toJSON()));
        restored.restart(() => {});
        battle.destroy();
        battle = restored;
    } else if (message.command === "label_root" && labelMode) {
        if (battle.ended || !battle.turn || !battle.sides.some(side => view(side)?.phase === "moving")) {
            throw new Error("A label root requires a pending Battle decision");
        }
        labelRoot = {battle: JSON.stringify(battle.toJSON()), revealed: structuredClone(revealed), logPosition};
    } else if (message.command === "label_restore" && labelMode && labelRoot) {
        const restored = Battle.fromJSON(labelRoot.battle);
        restored.restart(() => {});
        if (message.seed) restored.resetRNG(message.seed);
        battle.destroy();
        battle = restored;
        revealed.clear();
        for (const [key, known] of structuredClone(labelRoot.revealed)) revealed.set(key, known);
        logPosition = labelRoot.logPosition;
    } else {
        throw new Error("Unknown Showdown bridge command");
    }
    trackReveals();
    const views = battle.ended ? [] : battle.sides.map(view).filter(Boolean);
    let protocolLog;
    const publicLog = () => protocolLog ??= extractChannelMessages(battle.log.join("\n"), [0])[0].filter(
        line => !/^\|(debug|showteam|t:)\|/.test(line),
    );
    for (const item of views) {
        if (item.phase === "moving" && searchSeats.includes(item.seat)) {
            const {active, forceSwitch} = battle.getSide(item.seat).activeRequest;
            item.search = {
                events: searchEvents,
                force_switch: Boolean(forceSwitch?.some(Boolean)),
                own_pp: (active || []).map(slot => Object.fromEntries((slot?.moves || []).map(move => [move.id, move.pp]))),
            };
            if (proxySeats.includes(item.seat)) item.search.protocol_log = publicLog();
            if (proposalPreviews.has(item.seat)) {
                item.search.own_proxy = {
                    pivot: {request: battle.getSide(item.seat).activeRequest, preview_request: proposalPreviews.get(item.seat)},
                    templates: item.templates,
                };
            }
        }
    }
    const pivotView = views.find(item => item.seat === pivotSeat && item.phase === "moving");
    if (pivotView) {
        pivotView.pivot = {
            request: battle.getSide(pivotSeat).activeRequest,
            preview_request: pivotPreview,
            protocol_log: publicLog(),
        };
    }
    return {
        ended: battle.ended,
        winner: battle.winner || null,
        views,
    };
}

const processCommand = process.argv[2] === "--public-search" ? require("./search.cjs")({view, choiceText, movingOptions}) : handle;
readline.createInterface({input: process.stdin}).on("line", line => {
    try {
        process.stdout.write(JSON.stringify(processCommand(JSON.parse(line))) + "\n");
    } catch (error) {
        process.stdout.write(JSON.stringify({error: error.message}) + "\n");
    }
});
