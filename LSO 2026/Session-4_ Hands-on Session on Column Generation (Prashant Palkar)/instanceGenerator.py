import random

with open("items4.csv", "w") as f:
    f.write("item,length,demand\n")

    for i in range(1, 201):

        length = random.randint(10, 60)

        demand = random.randint(200, 1500)

        f.write(f"{i},{length},{demand}\n")
