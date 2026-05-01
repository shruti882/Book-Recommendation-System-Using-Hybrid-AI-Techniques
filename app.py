from flask import Flask, render_template, request
import pickle
import numpy as np

popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

app = Flask(__name__)


def ann_score(similarity_array):

    weights = np.ones(len(similarity_array)) * 0.5

    score = np.dot(similarity_array, weights)


    final_score = 1 / (1 + np.exp(-score))
    return final_score





def fuzzy_score(similarity):
    if similarity < 0.3:
        return "⚡ Try This"
    elif similarity < 0.7:
        return "👍 Recommended"
    else:
        return "⭐ Highly Recommended"



@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values))


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    if user_input not in pt.index:
        return render_template('recommend.html', data=[])

    index = np.where(pt.index == user_input)[0][0]
    distances = similarity_scores.iloc[index].values


    ann_result = ann_score(distances)

    similar_books = sorted(list(enumerate(distances)),
                           key=lambda x: x[1],
                           reverse=True)[1:6]

    data = []

    for i in similar_books:
        book_title = pt.index[i[0]]
        similarity_value = i[1]


        fuzzy_label = fuzzy_score(similarity_value)

        temp_df = books[books.iloc[:, 0].astype(str).str.strip().str.lower()
                        == book_title.strip().lower()]

        if not temp_df.empty:
            item = {
                "title": temp_df.iloc[0]['Book-Title'],
                "author": temp_df.iloc[0]['Book-Author'],
                "image": temp_df.iloc[0]['Image-URL-M'],
                "fuzzy": fuzzy_label,
                "ann": round(float(ann_result), 2)  # keep but not show
            }
        else:
            item = {
                "title": book_title,
                "author": "",
                "image": "",
                "fuzzy": fuzzy_label,
                "ann": round(float(ann_result), 2)
            }

        data.append(item)

    return render_template('recommend.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
