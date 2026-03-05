"""
This script file has been copied from the MPVR paper's original code repository from 'generate/ucf101.py' and moderately modified for this project.
MPVR Repository: https://github.com/jmiemirza/Meta-Prompting/blob/master/descriptions/gpt/UCF101.json

Modifications/Additions were made to generate descriptions for the ActivityNet Dataset instead of for the UCF101 dataset.
Improvements were also made to improve readability and maintainability.
"""

import json

import openai
from tqdm import tqdm

from src.utils.constants import DATA_DIR

openai.api_key = ""

all_json_dict = {}
all_responses = {}
root_folder = DATA_DIR
if not root_folder.is_dir():
    raise ValueError("Folder does not exist")
vowel_list = ["A", "E", "I", "O", "U"]

# fmt: off
activitynet_classes = ['River tubing','Horseback riding','Playing field hockey','Disc dog','Hanging wallpaper','Fun sliding down','Ballet','Changing car wheel','Waxing skis','Shoveling snow','Belly dance','Breakdancing','Knitting','Mooping floor','Spinning','Waterskiing','Arm wrestling','Bungee jumping','Rock climbing','Playing harmonica','Kneeling','Doing a powerbomb','Doing fencing','Playing polo','Preparing salad','Drum corps','Shot put','Ironing clothes','Shuffleboard','Swinging at the playground','Playing blackjack','Capoeira','Using the balance beam','Cricket','Playing water polo','Ice fishing','Preparing pasta','Making a sandwich','Mowing the lawn','Powerbocking','Hopscotch','Playing flauta','Braiding hair','Rafting','Drinking beer','Doing karate','Discus throw','Fixing bicycle','Long jump','Tumbling','Surfing','Elliptical trainer','Springboard diving','Slacklining','Playing ten pins','Tango','Installing carpet','Hammer throw','Decorating the Christmas tree','Using uneven bars','Washing hands','Canoeing','Rollerblading','Making a cake','Curling','Roof shingle removal','Plataform diving','Swimming','Starting a campfire','Throwing darts','Assembling bicycle','Kayaking','Running a marathon','Playing congas','Wrapping presents','Getting a tattoo','Tug of war','Croquet','Painting','Using the monkey bar','Playing pool','Chopping wood','Playing ice hockey','Clean and jerk','Raking leaves','Putting on makeup','Doing crunches','Playing rubik cube','Beer pong','Hitting a pinata','Doing step aerobics','Volleyball','Trimming branches or hedges','Beach soccer','Tennis serve with ball bouncing','Brushing hair','Snowboarding','Blow-drying hair','Skiing','Pole vault','Using the pommel horse','Mixing drinks','Vacuuming floor','Cheerleading','Having an ice cream','Using parallel bars','Bathing dog','Scuba diving','Layup drill in basketball','Hula hoop','Rope skipping','Doing motocross','Table soccer','Painting furniture','Grooming horse','Shaving legs','Using the rowing machine','Smoking hookah','Putting in contact lenses','Hurling','Playing beach volleyball','Riding bumper cars','Playing kickball','Washing dishes','Longboarding','Dodgeball','Calf roping','Making an omelette','Playing guitarra','Clipping cat claws','Cumbia','Polishing shoes','Getting a piercing','Painting fence','Drinking coffee','Removing curlers','Plastering','Doing nails','Brushing teeth','BMX','Hand car wash','Ping-pong','Snow tubing','Laying tile','Wakeboarding','Cleaning windows','Playing badminton','Sharpening knives','Grooming dog','Getting a haircut','Triple jump','Kite flying','High jump','Windsurfing','Rock-paper-scissors','Baton twirling','Zumba','Playing lacrosse','Walking the dog','Playing bagpipes','Building sandcastles','Sailing','Putting on shoes','Removing ice from car','Paintball','Sumo','Doing kickboxing','Shaving','Smoking a cigarette','Futsal','Carving jack-o-lanterns','Polishing forniture','Camel ride','Playing saxophone','Hand washing clothes','Playing squash','Playing accordion','Washing face','Welding','Applying sunscreen','Javelin throw','Tai chi','Fixing the roof','Peeling potatoes','Archery','Playing racquetball','Blowing leaves','Playing drums','Cleaning shoes','Spread mulch','Snatch','Baking cookies','Bullfighting','Cleaning sink','Playing piano','Playing violin','Skateboarding','Cutting the grass','Making a lemonade']
category_list_all = {"ActivityNet": activitynet_classes}
# fmt: on

for dataset_name, class_names in category_list_all.items():
    print("Generating descriptions for " + dataset_name + " dataset.")

    json_name_all = root_folder / f"{dataset_name}.json"

    if json_name_all.is_file():
        raise ValueError("File already exists")

    for i, category in enumerate(tqdm(class_names)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        # Dataset Name: ActivityNet
        # Description: ActivityNet is a large-scale video dataset for human activity understanding containing 200 different types of activities and over 10,000 untrimmed videos from YouTube. It covers tasks like global video classification, trimmed activity classification, and temporal activity detection.

        prompts = []
        # fmt: off
        # 1-5
        prompts.append("Describe what is happening in a video showing " + category + ".")
        prompts.append("What are the key visual cues that indicate the activity " + category + "?")
        prompts.append("How would you recognise the activity " + category + " in an untrimmed video?")
        prompts.append("Explain the sequence of actions typically involved in " + category + ".")
        prompts.append("What objects and people are usually present during " + category + "?")
        prompts.append("Describe the typical setting where " + category + " takes place.")
        prompts.append("How does the activity " + category + " usually begin and end in a video?")
        prompts.append("What movements distinguish " + category + " from similar activities?")
        prompts.append("Identify the temporal segments that correspond to " + category + " in a long video.")
        prompts.append("What makes " + category + " visually distinctive in a real world video?")
        prompts.append("Describe a realistic YouTube style video depicting " + category + ".")
        prompts.append("How can you temporally localise the activity " + category + " within a full length recording?")
        prompts.append("What human interactions are commonly seen in " + category + "?")
        prompts.append("Explain the pace and rhythm typically associated with " + category + ".")
        prompts.append("What background elements help confirm the activity " + category + "?")
        prompts.append("Describe the camera perspective often used when recording " + category + ".")
        prompts.append("How would you differentiate " + category + " from a closely related activity?")
        prompts.append("What are the defining stages of the activity " + category + "?")
        prompts.append("Describe the environment and context surrounding " + category + ".")
        prompts.append("What visual patterns repeatedly appear in videos of " + category + "?")
        prompts.append("How can motion patterns help identify " + category + "?")
        prompts.append("What types of participants are typically involved in " + category + "?")
        prompts.append("Describe a short clip that clearly illustrates " + category + ".")
        prompts.append("What temporal cues signal the transition into or out of " + category + "?")
        prompts.append("How would an annotation system label the start and end of " + category + "?")
        prompts.append("What challenges might arise when detecting " + category + " in unconstrained videos?")
        prompts.append("Describe the core action that defines " + category + ".")
        prompts.append("What contextual clues support accurate classification of " + category + "?")
        prompts.append("How does the surrounding scene contribute to recognising " + category + "?")
        prompts.append("Summarise the essential visual characteristics of the activity " + category + ".")
        # fmt: on

        res_ = {}

        for curr_prompt in prompts:
            all_result = []

            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=curr_prompt,
                temperature=0.99,
                max_tokens=50,
                n=10,
            )

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")

            res_[curr_prompt] = all_result

        all_responses[category] = res_
        with json_name_all.open("w") as f:
            json.dump(all_responses, f, indent=4)
