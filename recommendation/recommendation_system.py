def read_file(filename):
    users = []
    news = []
    with open(filename, 'r') as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip('\n').split('|')
            users.append(user)
            news.append(item)
    return users, news


def create_dictionary(users, news):
    users_dict = {}
    for user, item in zip(users, news):
        if user not in users_dict:
            users_dict[user] = list()
        users_dict[user].append(item)
    return users_dict

def make_user_interest_vector(user_interests):
    return [1 if interest in user_interests else 0 for interest in unique_interests]


users, news = read_file('viewed_news.csv')
users_interests = create_dictionary(users, news)
unique_interests = sorted(set(news))


user_interest_matrix = map(make_user_interest_vector, users_interests)
print(user_interest_matrix)
