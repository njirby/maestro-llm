"""Checker tests on narrations the probes actually produced (outputs/sounddiff/omni_fi/*.jsonl, probe1 targets; flags
from the verified notes in outputs/sounddiff/fi_lib/notes.json) plus delta claims."""
from maestro.sounddiff.check import check_delta, check_state

# probe1 f038 (instruct_notes4): one unbroken line, held - correct, must pass the continuity / length checks
F038 = {"dry": True, "no_dist": True, "steady": True, "pitch_steady": True, "single": True, "bright": True,
        "short_notes": True, "low": True, "continuous": True}
F038_TEXT = ("This feels like a droning, tension-building pad underneath a track rather than an active melody. The notes "
             "are held seamlessly in a dreamlike, unbroken line, neither shifting nor rippling; they just press forward "
             "with quiet insistence in the center of the stereo field.")
# probe1 f004: plucky, separated, high single line
F004 = {"dry": True, "no_dist": True, "steady": True, "pitch_steady": True, "wide": True, "single": True,
        "plucky": True, "short_notes": True, "separated": True, "high": True}
# probe1 f029: plucky chords
F029 = {"chords": True, "plucky": True}
# probe1 f011: dark, held chords that run into each other
F011 = {"dry": True, "no_dist": True, "pitch_steady": True, "wide": True, "chords": True, "dark": True,
        "continuous": True, "high": True}
# probe1 f031: dark, high, held
F031 = {"pitch_steady": True, "narrow": True, "chords": True, "dark": True, "held": True, "continuous": True,
        "high": True}


def test_continuous_text_passes_on_continuous_part():
    assert check_state(F038_TEXT, F038) == []


def test_continuous_claim_flagged_on_separated_part():
    t = ("A crisp little pluck line in the upper midrange. The notes blend into each other in one seamless legato "
         "stream.")
    assert "notes running together when separated" in check_state(t, F004)


def test_gaps_claim_flagged_on_continuous_part():
    t = ("A low, buzzy single line with clean breaks between every note; it sounds choppy and detached overall, very "
         "bright.")
    assert "silence between notes when they run together" in check_state(t, F038)


def test_pad_label_on_plucky_part():  # instruct_notes4 f029
    t = ("This is a atmospheric synth pad with a hypnotic ripple—notes hang with a gently plucked feel, each one "
         "pulsing with subtle internal brightness shifting.")
    assert "'pad' on a plucky part" in check_state(t, F029)


def test_bright_claim_on_dark_sound():  # instruct_p f011
    t = ("Okay, this snippet plays a descending melody in a mid-to-high register. Overall, the tone is bright and clear "
         "with a sterile, electronic chime-like purity, almost like a metallic bell.")
    assert "bright on a dark sound" in check_state(t, F011)


def test_dark_claim_on_dark_sound_passes():  # instruct_notes4 f011
    t = ("This is a very wide, immersive part. The tone is dark and luminous, with a gentle swell as each chord holds, "
         "creating a feeling of depth.")
    assert "bright on a dark sound" not in check_state(t, F011)


def test_brightness_noun_and_comparative_not_claims():
    t = ("A soft, held chord part with dark tone; the brightness never jumps, and nothing gets brighter over time, "
         "it just sits there.")
    assert check_state(t, F011) == []


def test_negated_brightness_not_flagged():
    t = "A dark, held chord part, not bright at all and without any sparkle, sitting wide and still in the mix."
    assert "bright on a dark sound" not in check_state(t, F011)


def test_bass_heavy_is_not_a_register_claim():  # instruct_p f031: 'bass-heavy' is tone, not the 'bass' label
    t = ("That's a powerful, bass-heavy sound in a high register; dark and full, with chords that hold and stay "
         "centered.")
    assert "'bass' on a high part" not in check_state(t, F031)


def test_bassline_on_high_part_flagged():
    t = "A punchy little bassline: dark, held chords that stay narrow and centered, steady pitch throughout the part."
    assert "'bass' on a high part" in check_state(t, F031)


def test_sustained_on_plucky_part():  # instruct_p f004-style claim
    t = ("It's a higher-register single line with sustained notes that ring out, dry and wide, steady in pitch, with "
         "no distortion.")
    assert "sustained/held on a plucky part" in check_state(t, F004)


def test_lingering_room_is_not_note_length():  # instruct_notes4 f010
    t = ("It's a stark, plucked melody. There's a touch of lingering room around it where the sound breathes, giving "
         "it just a hint of shape.")
    assert "sustained/held on a plucky part" not in check_state(t, {"plucky": True})


def test_pluck_label_on_held_notes():
    t = "Heavy, dark chords that pluck along in the upper register, narrow and centered, steady pitch."
    assert "plucky/staccato on held notes" in check_state(t, F031)


def test_reverb_on_dry_clip_and_negation():
    t1 = "A plucky, high single line drenched in reverb, wide and steady with no distortion at all here."
    t2 = "A plucky, high single line with no reverb or delay, wide and steady with no distortion at all here."
    assert "reverb/echo on a dry clip" in check_state(t1, F004)
    assert check_state(t2, F004) == []


def test_short_and_non_english():
    assert "too short / empty" in check_state("Nice pluck.", F004)
    assert "non-English text" in check_state("A plucky high line that sounds dry and wide 很好 and steady overall.",
                                             F004)


# ------------------------------------------------------------------------------------------------ deltas
SE = {"mode": "self_edit", "overall": "strong", "dirs": {"brightness": 1, "level": -1, "space": 0, "width": 0,
                                                          "movement": 0, "distortion": 0, "attack": 0},
      "flags": {"no_dist": True}}


def test_self_edit_correct_text_passes():
    t = ("That wavetable swap made it way brighter and buzzier, with a lot more sparkle on top, though it also came "
         "out quieter. Space and movement didn't change.")
    assert check_delta(t, SE) == []


def test_self_edit_wrong_direction():
    t = "That made the whole thing darker and warmer, and a bit louder too, which sits nicely in the mix."
    p = check_delta(t, SE)
    assert "claims brightness down, measured up" in p
    assert "claims level up, measured down" in p


def test_self_edit_same_level_claim():  # Omni on probe1 e001: 'the level is still pretty quiet'
    t = ("The movement is in exactly the same place and the level is still pretty quiet overall, but the sparkle is "
         "totally different now, much brighter.")
    assert "says level unchanged, measured down" in check_delta(t, SE)


def test_self_edit_no_change_on_audible_edit():
    t = "Honestly, I can't hear any difference at all between those two, it sounds exactly the same to me."
    assert "says no change on an audible edit" in check_delta(t, SE)


def test_self_edit_change_claim_on_inaudible_edit():
    d = {"mode": "self_edit", "overall": "none", "dirs": {k: 0 for k in SE["dirs"]}, "flags": {}}
    assert check_delta("Honestly I can't hear a difference between before and after, it sounds the same.", d) == []
    assert "claims space changed on an inaudible edit" in check_delta(
        "That added a lot more reverb, the tail is longer now and it sits further back in the mix.", d)


def test_self_edit_distortion_state_check():
    t = "Way brighter now, and the new distortion adds a nice crunch, a bit quieter overall than before though."
    assert "state: distortion when none" in check_delta(t, SE)


TVC = {"mode": "target_vs_current", "overall": "clear", "trajectory": "same",
       "dirs": {"brightness": -1, "level": 0, "space": 1}, "flags": {}}


def test_tvc_direction_relative_to_subject():
    ok = "Not much closer yet. The target is darker than mine and has more reverb, so I'll go after the tone."
    bad = "Not much closer yet. The target is brighter than mine, so I need to darken my patch a little more."
    assert check_delta(ok, TVC) == []
    assert "claims brightness up, measured down" in check_delta(bad, TVC)
    # the same claim made about 'my patch' flips: mine is brighter = the target is darker (correct)
    assert check_delta("Not much closer. My patch is brighter than the target and the target has more reverb.",
                       TVC) == []


def test_tvc_trajectory_and_match():
    assert "says closer, measured same" in check_delta(
        "That got me closer; the target is still darker with more reverb than my patch.", TVC)
    assert "says it matches the target when clearly different" in check_delta(
        "Nailed it, that's spot on now and there's nothing left to change at all here.", TVC)


# probe1 self_edit Omni outputs the first delta checker got wrong (false positives)
def test_everything_else_same_is_not_no_change():  # e003
    t = ("The main thing that changed is the attack: now it's much softer and more drawn out. Everything else in there "
         "sounds pretty much the same.")
    assert "says no change on an audible edit" not in check_delta(t, SE)


def test_cleaner_hiss_is_not_distortion():  # e002 ('Noise: cleaner, less hiss' in the notes)
    t = "The most noticeable change is the added reverb. It's also a bit quieter and cleaner, with the hiss dialed back."
    d = {**SE, "dirs": {**SE["dirs"], "space": 1}}
    assert not [p for p in check_delta(t, d) if "distortion" in p]


def test_space_between_is_not_reverb():  # e016
    t = ("The wobble in the brightness feels a bit slower now; there's a bit more space between the shifts, giving the "
         "movement a smoother pace. Brighter, quieter.")
    assert not [p for p in check_delta(t, SE) if "space" in p]


# probe1 target_vs_current Omni outputs the first checker misread
TVC2 = {"mode": "target_vs_current", "overall": "strong", "trajectory": "same",
        "dirs": {"brightness": -1, "level": -1, "distortion": 1}, "flags": {}}


def test_didnt_get_closer_is_negated():  # t000
    t = ("Nope, that edit didn't really get us closer. The biggest thing still missing is that the target has this "
         "heavier, grittier distortion to it.")
    assert check_delta(t, TVC2) == []


def test_goal_to_get_closer_is_not_a_claim():  # t022
    t = "Not much change from that edit. The target is darker; I need to dial down the brightness to get closer."
    assert check_delta(t, TVC2) == []


def test_further_in_the_mix_is_not_trajectory():  # t045
    t = ("The edit didn't change much. The target has a brighter feel, while the target sits a bit further away in "
         "the mix.")
    assert not [p for p in check_delta(t, TVC2) if "further" in p]


def test_drifted_further_is_trajectory():
    t = "That edit drifted further from the target; the target is darker and grittier than mine right now."
    assert "says further, measured same" in check_delta(t, TVC2)


def test_dimension_spot_on_is_not_a_match_claim():  # t045
    t = "Not much closer, but the movement is spot on now; the target is still darker and grittier than my patch."
    assert "says it matches the target when clearly different" not in check_delta(t, TVC2)


def test_possessive_target_is_subject():  # t058
    t = "The biggest thing left is that my patch still feels too upfront compared to the target's darker quality."
    assert not [p for p in check_delta(t, TVC2) if "brightness" in p]


def test_interjection_opener_flagged():  # WS-A rollouts: 27B edit narrations opening with "Yeah,"
    for t in ("Yeah, that last tweak was clearly audible and pulled me closer; the target is still darker than mine.",
              "Okay, so the main change is that it got much brighter and quieter, the space stayed the same though.",
              "Alright, that made it a lot brighter and a bit quieter, everything else sits where it was before.",
              "Hey, good news: it's way brighter now and a bit quieter, with the movement exactly where it was."):
        assert "interjection opener" in check_delta(t, SE)
    assert "interjection opener" not in check_delta(
        "Much brighter now, and a bit quieter; yeah, the space and movement stay where they were before.", SE)
    assert "interjection opener" not in check_delta(
        "Sounds way brighter now and a bit quieter, with the space and the movement right where they were.", SE)


# v5 env rounds (narr_defer1b slots 04 / 06): a longer release on a dry patch is ring-out, not space
ENV = {"mode": "self_edit", "overall": "clear", "dirs": {"space": 0, "tail": 1, "attack": 0, "movement": 0},
       "flags": {"dry": True, "no_dist": True, "steady": True}}


def test_ring_out_is_not_space():
    t = ("The notes now ring out longer after they're released, a softer, longer release on each one, while the space "
         "stays completely dry.")
    assert check_delta(t, ENV) == []


def test_echo_on_release_edit_flagged():  # slot 06 narration
    t = ("A little more echo is hanging around the notes now, and the held notes fade gently instead of staying "
         "solid.")
    p = check_delta(t, ENV)
    assert "state: reverb/echo on a dry clip" in p and "claims space up, measured unchanged" in p


def test_negated_list_of_dimensions():  # slot 06 narrations after the 'unchanged' wording
    d = {"mode": "self_edit", "overall": "clear", "dirs": {k: 0 for k in ("brightness", "level", "space", "distortion",
                                                                          "movement")}, "flags": {"dry": True}}
    for t in ("The part just has a slightly different color to it now, and I can hear it, but it isn't sitting louder, "
              "brighter, grittier, or more open.",
              "The sound's color has shifted a little. It's not louder, brighter, more echoey, grittier, or moving "
              "differently, just a subtle change in the tone itself.",
              "Its overall tone has a slight colour shift that changes the character without making it louder, "
              "brighter, or more spacious."):
        assert check_delta(t, d) == [], t
    # a list item after a non-negated opener is still a claim
    assert check_delta("It got much louder, brighter, and a lot more echoey than before, a really big change.", d)


def test_hold_steadier_is_note_length_not_movement():  # v5 narr_defer2 t002_000/self_edit/10
    d = {"mode": "self_edit", "overall": "strong", "dirs": {"level": 1, "brightness": 1, "movement": 1, "sustain": 1,
                                                             "space": 0}, "flags": {"dry": True}}
    t = ("It's way louder and much brighter, with a buzzy edge, and the brightness keeps shifting while each note plays. "
         "The notes hold steadier and a little longer instead of fading, with a subtle wobble in the brightness. It "
         "sounds more alive, but the dryness and the way the notes come in still feel the same.")
    assert check_delta(t, d) == []
    assert "claims movement down, measured up" in check_delta(
        "Much louder and brighter now, but the tone is steadier, with the wobble calmed right down.", d)


def test_hang_steadier_is_note_length():  # pilot20c s0003/self_edit/06
    d = {"mode": "self_edit", "overall": "clear", "dirs": {"movement": 1, "attack": 1}, "flags": {}}
    t = ("Each note now swells and dips instead of sitting steady. The attack feels softer and a bit slower, while the "
         "notes hang steadier after that.")
    assert check_delta(t, d) == []


def test_echo_free_is_a_negation():  # pilot20c s0002/self_edit/17
    d = {"mode": "self_edit", "overall": "clear", "dirs": {"brightness": 1}, "flags": {"dry": True, "no_dist": True}}
    t = "It's a touch brighter now. The rest of the plucky, echo-free character is unchanged, still distortion-free."
    assert check_delta(t, d) == []
    assert "state: reverb/echo on a dry clip" in check_delta(
        "It's a touch brighter now, and a little echo trails each note after the edit.", d)
