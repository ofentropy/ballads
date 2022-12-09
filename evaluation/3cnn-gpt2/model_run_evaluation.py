from model_evaluation_metrics import *

ballads, related_words, prompt_rhymes = json_parser("https://raw.githubusercontent.com/nmailan/cs230ballad/main/generated_ballads (1).json")

#SYLLABLES EVALUATION
rate = eval_syllables_across_ballads(ballads)
print("Syllables evaluation average: ", rate)

#RHYME EVALUATION
rateABAB, rateABCB, rateABAC, rateAABB, overall_rhyme = eval_follows_rhyme_scheme(ballads,prompt_rhymes)
print("ABAB evaluation average: ", rateABAB)
print("ABCB evaluation average: ", rateABCB)
print("ABAC evaluation average: ", rateABAC)
print("AABB evaluation average: ", rateAABB)
print("Overall rhyme evaluation average: ", overall_rhyme)

#CONCEPTNET RELATEDNESS EVALUATION
relatedness_score = get_average_ballad_relatedness_score(ballads, related_words)
print("Relatedness score evaluation average: ", relatedness_score)
