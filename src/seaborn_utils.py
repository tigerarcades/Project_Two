def draw_count_plot(plt, sns, data_frame, column_name):
    f, ax = plt.subplots(figsize=(20, 10))
    sns.countplot(data=data_frame, x=column_name)
    plt.xticks(rotation=45)
