import csv
import os

# Open the CSV file in read mode and create a list of rows
with open('output.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

# Insert a new value in the second column of each row
folder_path = r'C:\Users\dingln51075\Desktop\Intern\labelme\images_with_label'
for row,f in zip(rows, os.listdir(folder_path)):
    row.insert(4, f)

# Write the updated rows to a new CSV file
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)