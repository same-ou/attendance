from datetime import datetime

def mark_attendance(name, attendance_path):
    with open(attendance_path, 'r+') as f:
        my_data_list = f.readlines()
        name_list = [line.split(',')[0] for line in my_data_list]
        if name not in name_list:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')
