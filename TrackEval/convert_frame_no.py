import os

def increment_frame_numbers(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        # Split the line by commas and strip any extra whitespace
        parts = line.strip().split(',')

        # Increment the first part (frame number) by 1, leave the rest unchanged
        #if parts[0].isdigit():  # Make sure the first part is indeed a number
        frame_number = int(parts[0])+1
        tracking_id = parts[1]
        x1 = parts[2]
        y1 = parts[3]
        width = parts[4]
        height = parts[5]
        confidence = parts[6]

        # Join the parts back together with commas
        new_line = f"{frame_number},{tracking_id},{x1},{y1},{width},{height},{confidence}, -1, -1, -1"
        updated_lines.append(new_line)

    #filename = 'data/trackers/mot_challenge/MOT20-test/Snow_King_Mountain_Lifts_2024-02-16_19_05_430_622/gt/gt.txt'
    # Write the updated content back to the file
    with open(filename, 'w') as file:
        for line in updated_lines:
            file.write(line + '\n')
        #file.writelines(updated_lines)


# Usage
folder='data/trackers/mot_challenge/MOT20-test/yolov11_bot/data'
for txt in os.listdir(folder):
    filename= os.path.join(folder, txt)
    increment_frame_numbers(filename)
#filename = 'data/gt/mot_challenge/MOT20-test/Snow_King_Mountain_Lifts_2024-02-16_19_05_430_622/gt/Snow_King_Mountain_Lifts_2024-02-16_19_05_430_622.txt'  # Replace with your file path
#increment_frame_numbers(filename)

#Snow_King_Mountain_Lifts_2024-02-15_20_43_20315_20435