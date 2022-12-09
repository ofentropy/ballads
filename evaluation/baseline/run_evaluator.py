from evaluation_metrics_baseline import *

ballads, related_words = json_parser("https://raw.githubusercontent.com/nmailan/cs230ballad/main/generated_ballads.json")
ballads_aws, related_words_aws = json_parser("https://raw.githubusercontent.com/ofentropy/balladsfiles/main/generated_ballads.json")
ballads_colab, related_words_colab = json_parser("https://raw.githubusercontent.com/ofentropy/balladsfiles/main/generated_ballads_colab.json")

syl_rate = eval_syllables_across_ballads(ballads)
syl_rate_aws = eval_syllables_across_ballads(ballads_aws)
syl_rate_colab = eval_syllables_across_ballads(ballads_colab)

print("Syllables evaluation average: ", syl_rate)
print("Syllables evaluation average (aws): ", syl_rate_aws)
print("Syllables evaluation average (colab): ", syl_rate_colab)

rateABAB, rateABCB, rateABAC, rateAABB, overall_rhyme = eval_follows_rhyme_scheme(ballads)
rateABAB_aws, rateABCB_aws, rateABAC_aws, rateAABB_aws, overall_rhyme_aws = eval_follows_rhyme_scheme(ballads_aws)
rateABAB_colab, rateABCB_colab, rateABAC_colab, rateAABB_colab, overall_rhyme_colab = eval_follows_rhyme_scheme(ballads_colab)

print("ABAB evaluation average: ", rateABAB)
print("ABCB evaluation average: ", rateABCB)
print("ABAC evaluation average: ", rateABAC)
print("AABB evaluation average: ", rateAABB)
print("Overall rhyme evaluation average: ", overall_rhyme)

print("ABAB evaluation average _aws : ", rateABAB_aws)
print("ABCB evaluation average _aws : ", rateABCB_aws)
print("ABAC evaluation average _aws : ", rateABAC_aws)
print("AABB evaluation average _aws : ", rateAABB_aws)
print("Overall rhyme evaluation average _aws : ", overall_rhyme_aws)

print("ABAB evaluation average _colab : ", rateABAB_colab)
print("ABCB evaluation average _colab : ", rateABCB_colab)
print("ABAC evaluation average _colab : ", rateABAC_colab)
print("AABB evaluation average _colab : ", rateAABB_colab)
print("Overall rhyme evaluation average _colab : ", overall_rhyme_colab)

relatedness_score = get_average_ballad_relatedness_score(ballads, related_words)
relatedness_score_aws = get_average_ballad_relatedness_score(ballads_aws, related_words_aws)
relatedness_score_colab = get_average_ballad_relatedness_score(ballads_colab, related_words_colab)

print("Relatedness score evaluation average: ", relatedness_score)
print("Relatedness score evaluation average aws: ", relatedness_score_aws)
print("Relatedness score evaluation average colab: ", relatedness_score_colab)
