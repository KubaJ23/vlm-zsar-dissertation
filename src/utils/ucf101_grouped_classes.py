import collections
import pprint

import pandas as pd

from src.utils.constants import TEST_CSV

ucf101_categories_camelcase = {
    "Body-Motion Only": [
        "BabyCrawling",
        "BlowingCandles",
        "BodyWeightSquats",
        "HandstandPushups",
        "HandstandWalking",
        "JumpingJack",
        "Lunges",
        "PullUps",
        "PushUps",
        "RockClimbingIndoor",
        "RopeClimbing",
        "Swing",
        "TaiChi",
        "TrampolineJumping",
        "WalkingWithDog",
        "WallPushups",
    ],
    "Human-Human Interaction": [
        "BandMarching",
        "Haircut",
        "HeadMassage",
        "MilitaryParade",
        "SalsaSpin",
    ],
    "Human-Object Interaction": [
        "ApplyEyeMakeup",
        "ApplyLipstick",
        "BlowDryHair",
        "BrushingTeeth",
        "CuttingInKitchen",
        "Hammering",
        "HulaHoop",
        "JugglingBalls",
        "JumpRope",
        "Knitting",
        "Mixing",
        "MoppingFloor",
        "Nunchucks",
        "PizzaTossing",
        "ShavingBeard",
        "SkateBoarding",
        "SoccerJuggling",
        "Typing",
        "WritingOnBoard",
        "YoYo",
    ],
    "Playing Musical Instruments": [
        "Drumming",
        "PlayingCello",
        "PlayingDaf",
        "PlayingDhol",
        "PlayingFlute",
        "PlayingGuitar",
        "PlayingPiano",
        "PlayingSitar",
        "PlayingTabla",
        "PlayingViolin",
    ],
    "Sports": [
        "Archery",
        "BalanceBeam",
        "BaseballPitch",
        "Basketball",
        "BasketballDunk",
        "BenchPress",
        "Biking",
        "Billiards",
        "Bowling",
        "BoxingPunchingBag",
        "BoxingSpeedBag",
        "BreastStroke",
        "CleanAndJerk",
        "CliffDiving",
        "CricketBowling",
        "CricketShot",
        "Diving",
        "Fencing",
        "FieldHockeyPenalty",
        "FloorGymnastics",
        "FrisbeeCatch",
        "FrontCrawl",
        "GolfSwing",
        "HammerThrow",
        "HighJump",
        "HorseRace",
        "HorseRiding",
        "IceDancing",
        "JavelinThrow",
        "Kayaking",
        "LongJump",
        "ParallelBars",
        "PoleVault",
        "PommelHorse",
        "Punch",
        "Rafting",
        "Rowing",
        "Shotput",
        "Skiing",
        "Skijet",
        "SkyDiving",
        "SoccerPenalty",
        "StillRings",
        "SumoWrestling",
        "Surfing",
        "TableTennisShot",
        "TennisSwing",
        "ThrowDiscus",
        "UnevenBars",
        "VolleyballSpiking",
    ],
}


def get_unique_labels_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        if "label" not in df.columns:
            print(f"Error: Column 'label' not found in {filepath}")
            return set()
        unique_labels = set(df["label"].unique())
        return unique_labels

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return set()
    except pd.errors.EmptyDataError:
        print(f"Error: File {filepath} is empty.")
        return set()
    except Exception as e:
        print(f"An error occurred: {e}")
        return set()


def check_dictionary_integrity(categories_dict):
    """
    Checks the dictionary for internal duplicates and returns a set of unique action labels.
    """
    all_actions = []
    for action_list in categories_dict.values():
        all_actions.extend(action_list)

    unique_actions = set(all_actions)

    if len(all_actions) != len(unique_actions):
        print("WARNING: Duplicates found within the dictionary definitions.")
        counter = collections.Counter(all_actions)
        duplicates = [item for item, count in counter.items() if count > 1]
        print(f"Duplicate classes: {duplicates}")
    else:
        print("SUCCESS: No duplicate classes found within the dictionary.")

    print(f"Dictionary contains {len(unique_actions)} unique classes.")
    return unique_actions


"""Run this file as a script to check the integrity of the dictionary against the test CSV file."""
if __name__ == "__main__":
    dict_labels = check_dictionary_integrity(ucf101_categories_camelcase)

    print(f"Attempting to read labels from: {TEST_CSV}\n")

    csv_labels = get_unique_labels_from_csv(TEST_CSV)

    if not csv_labels:
        print(f"TEST FAILED: Could not read labels from CSV at {TEST_CSV}.")
    else:
        print(f"CSV file contains {len(csv_labels)} unique classes.")

        extra_in_dict = dict_labels - csv_labels
        missing_from_dict = csv_labels - dict_labels

        if not extra_in_dict and not missing_from_dict:
            print("\nTEST RESULT: SUCCESS")
            print(f"Dictionary and CSV contain the same {len(dict_labels)} classes.")
        else:
            print("\nTEST RESULT: MISMATCH FOUND")

            if extra_in_dict:
                print(
                    f"\n{len(extra_in_dict)} classes are in the DICTIONARY but NOT in the CSV:"
                )
                pprint.pprint(sorted(list(extra_in_dict)))

            if missing_from_dict:
                print(
                    f"\n{len(missing_from_dict)} classes are in the CSV but NOT in the DICTIONARY:"
                )
                pprint.pprint(sorted(list(missing_from_dict)))
