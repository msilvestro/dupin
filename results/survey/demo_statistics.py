"""Perform some statistics on the demographics of the survey."""
# pylint: disable=C0103
import csv
import matplotlib.pyplot as plt  # to display plots

genders = {
    'male': 0,
    'female': 0
}
ages = {
    'first': 0,
    'second': 0,
    'third': 0,
    'fourth': 0,
    'fifth': 0,
    'sixth': 0
}
with open(
    'results/survey/results_demo.csv', 'r', encoding='utf8'
    ) as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_it = iter(csv_reader)
    next(csv_it)
    for user in csv_it:
        gender, age = user
        genders[gender] += 1
        ages[age] += 1

print(genders)
print(ages)
plt.bar(range(len(ages)), ages.values())
plt.xticks(range(len(ages)), ['<18', '18-24', '25-34', '35-44', '45-56', '>57'])
# plt.title("Age distribution of survey users")
plt.savefig('export/survey_age.pdf')
plt.show()
