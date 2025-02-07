import csv

# Define the path to the CSV file
csv_file = './csv/test.csv'

# Initialize an empty dictionary to store the mapping
species_mapping = {}

# Read the CSV file and populate the dictionary
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        species_name = row[1]  # Assuming species name is in the first column
        number = row[7]  # Assuming associated number is in the second column
        # Check if the species name and number are different
        if number not in species_mapping:
            species_mapping[number] = species_name

print(species_mapping)