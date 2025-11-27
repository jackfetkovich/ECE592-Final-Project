    filename = './data/command_vs_mpac_outputreelnew.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['time','x_goal', 'x', 'y_goal', 'y', 'u_v', 'v', 'u_w', 'w'])

    
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['time','x_goal', 'x', 'y_goal', 'y', 'u_v', 'v', 'u_w', 'w'])