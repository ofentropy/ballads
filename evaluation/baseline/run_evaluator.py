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

rateABAB, rateABCB, overall_rhyme = eval_follows_rhyme_scheme(ballads)
rateABAB_aws, rateABCB_aws, overall_rhyme_aws = eval_follows_rhyme_scheme(ballads_aws)
rateABAB_colab, rateABCB_colab, overall_rhyme_colab = eval_follows_rhyme_scheme(ballads_colab)

print("ABAB evaluation average: ", rateABAB)
print("ABCB evaluation average: ", rateABCB)
print("Overall rhyme evaluation average: ", overall_rhyme)

print("ABAB evaluation average _aws : ", rateABAB_aws)
print("ABCB evaluation average _aws : ", rateABCB_aws)
print("Overall rhyme evaluation average _aws : ", overall_rhyme_aws)

print("ABAB evaluation average _colab : ", rateABAB_colab)
print("ABCB evaluation average _colab : ", rateABCB_colab)
print("Overall rhyme evaluation average _colab : ", overall_rhyme_colab)

relatedness_score = get_average_ballad_relatedness_score(ballads, related_words)
relatedness_score_aws = get_average_ballad_relatedness_score(ballads_aws, related_words_aws)
relatedness_score_colab = get_average_ballad_relatedness_score(ballads_colab, related_words_colab)

print("Relatedness score evaluation average: ", relatedness_score)
print("Relatedness score evaluation average aws: ", relatedness_score_aws)
print("Relatedness score evaluation average colab: ", relatedness_score_colab)
