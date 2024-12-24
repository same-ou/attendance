from datetime import datetime

# Track attendance for the session to avoid duplicate entries
marked_attendance = set()

def mark_attendance(name, attendance_path):
    # Check if the name has already been marked for attendance in this session
    if name in marked_attendance:
        return  # Skip if the name is already marked

    # Read the current attendance data
    with open(attendance_path, 'r+') as f:
        my_data_list = f.readlines()
        name_list = [line.split(',')[0] for line in my_data_list]

        # Check if the name is already in the attendance list (global check)
        if name not in name_list:
            # Mark the attendance
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')
            # Add the name to the session's marked attendance list
            marked_attendance.add(name)
            print(f"Attendance marked for {name} at {time} on {date}")

