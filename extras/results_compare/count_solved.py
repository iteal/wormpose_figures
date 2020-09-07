import sys

input_log = sys.argv[1]

with open(input_log, "r") as f:
    lines = f.readlines()

wp_solved = 0
total = 0
tierpsy_solved = 0

for line in lines:
    numbers = [int(s) for s in line.split() if s.isdigit()]
    wp_solved += numbers[0]
    total += numbers[1]
    tierpsy_solved += numbers[2]

name = input_log[:-4]

print(f"For: {name} (total frames:{total}), tierpsy solved {tierpsy_solved} ({100*tierpsy_solved / total:.1f}%), wormpose solved {wp_solved} ({100*wp_solved / total:.1f}%) or {100*(wp_solved - tierpsy_solved)/total:.1f}% more than tierpsy")

