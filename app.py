from flask import Flask, render_template, request
import joblib
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('student_performance_model.pkl')
except FileNotFoundError:
    print("Error: 'student_performance_model.pkl' not found. Make sure the model file is in the same directory as app.py.")
    model = None

@app.route('/')
def home():
    """Renders the home page with no initial prediction or graphs."""
    return render_template('index.html', prediction=None, graph_html=None)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request, processes input data,
    generates predictions, and creates a Plotly pie chart for scores.
    """
    if model is None:
        return render_template('index.html', prediction="Error: Model not loaded. Please check server logs.", graph_html=None)

    try:
        # Retrieve numerical scores from the simplified form
        math = int(request.form.get('math'))
        read = int(request.form.get('reading'))
        write = int(request.form.get('writing'))

        # Define mapping for categorical features (used for default values)
        mapping = {
            'female': 0, 'male': 1,
            'group a': 0, 'group b': 1, 'group c': 2, 'group d': 3, 'group e': 4,
            "some high school": 0, "high school": 1, "some college": 2,
            "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5,
            'standard': 0, 'free/reduced': 1,
            'none': 0, 'completed': 1
        }

        # Provide default values for categorical features, as they are no longer in the form
        # These defaults are chosen to be common/neutral values.
        default_gender = mapping.get('female')
        default_race = mapping.get('group c')
        default_parent_edu = mapping.get('some college')
        default_lunch = mapping.get('standard')
        default_test_prep = mapping.get('none')

        # Create a DataFrame for the new data, using user inputs for scores and defaults for categorical
        new_data = pd.DataFrame([[default_gender, default_race, default_parent_edu, default_lunch, default_test_prep, math, read, write]],
                                columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                                         'test preparation course', 'math score', 'reading score', 'writing score'])

        # Make prediction
        result = model.predict(new_data)[0]
        prediction_text = "Passed" if result == 1 else "Failed" # Assuming 1 for passed, 0 for failed

        # --- Generate Student Score Pie Chart ---
        scores = [math, read, write]
        labels = ['Math Score', 'Reading Score', 'Writing Score']
        colors = ['#00e5ff', '#00b8d4', '#0099cc'] # Neon blue shades

        fig = go.Figure(data=[
            go.Pie(labels=labels, values=scores, hole=0.5,
                   marker_colors=colors,
                   textinfo='percent+label', # Show percentage and label on slices
                   hoverinfo='label+value+percent' # Show label, value, and percentage on hover
                  )
        ])
        fig.update_layout(
            title_text='Your Performance Breakdown',
            title_font_color='#00e5ff',
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background for the entire plot area
            plot_bgcolor='rgba(0,0,0,0)', # Transparent background for the plotting area
            font=dict(color='#e0e0e0'), # General font color for labels etc.
            showlegend=True, # Show legend
            legend=dict(font=dict(color='#00e5ff'), orientation="h", xanchor="center", x=0.5), # Legend font color, horizontal, centered
            margin=dict(l=20, r=20, t=80, b=20), # Adjust margins
            height=400 # Adjust height for better fit
        )
        graph_html = pio.to_html(fig, full_html=False, config={'displayModeBar': False}) # Hide modebar


        return render_template('index.html', prediction=prediction_text, graph_html=graph_html)

    except ValueError:
        return render_template('index.html', prediction="Error: Please enter valid numerical scores (0-100).", graph_html=None)
    except Exception as e:
        return render_template('index.html', prediction=f"An unexpected error occurred: {str(e)}", graph_html=None)

if __name__ == '__main__':
    # Ensure 'data' directory exists (though student_data.csv is no longer used for distribution chart)
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    app.run(debug=True)
