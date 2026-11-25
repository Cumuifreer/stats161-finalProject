"use strict";

// A local Showdown battle, driven through newline-delimited JSON over stdio.
// Only each side's pending choice and its player-visible board cross the bridge.
const readline = require("node:readline");
const {Battle, Teams} = require("pokemon-showdown");
const {extractChannelMessages} = require("pokemon-showdown/dist/sim/battle");

let battle;
let logPosition;
const revealed = new Map();
const previewRosters = new Map();

function pokemonKey(name) {
    return name.replace(/^(p[12])[ab]: /, "$1: ");
}

function trackReveals() {
    const messages = extractChannelMessages(battle.log.slice(logPosition).join("\n"), [0])[0];
    logPosition = battle.log.length;
    for (const line of messages) {
        const [, event, actor, detail, ...tags] = line.split("|");
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
    } else if (message.command === "choose") {
        const side = battle.getSide(message.seat);
        if (!battle.choose(side.id, choiceText(side, message.action))) {
            throw new Error(side.choice.error || "Showdown rejected the choice");
        }
    } else {
        throw new Error("Unknown Showdown bridge command");
    }
    trackReveals();
    return {
        ended: battle.ended,
        winner: battle.winner || null,
        views: battle.ended ? [] : battle.sides.map(view).filter(Boolean),
    };
}

readline.createInterface({input: process.stdin}).on("line", line => {
    try {
        process.stdout.write(JSON.stringify(handle(JSON.parse(line))) + "\n");
    } catch (error) {
        process.stdout.write(JSON.stringify({error: error.message}) + "\n");
    }
});
