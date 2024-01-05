from flask import Flask, request, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import re
from google.generativeai.types.generation_types import StopCandidateException

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCfsGhwvyhcjxFvE1_HDG8-OjbEYJQwbvg'

app = Flask(__name__)

G_llm = ChatGoogleGenerativeAI(model="gemini-pro")

prompt_temp = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'bloodType', 'medicalcondition', 'sleepduration',
                     'qualityofsleep', 'physicalactivitylevel'],
    template="Health Recommendation System:\n"
             "I want you to provide health, food, and exercise recommendations, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person BloodType: {bloodtype}\n"
             "Person MedicalCondition: {medicalcondition}\n"
             "Person SleepDuration: {sleepduration}\n"
             "Person QualityofSleep: {qualityofsleep}\n"
             "Person PhysicalActivityLevel: {physicalactivitylevel}\n"
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if request.method == 'POST':
            age = request.form['age']
            gender = request.form['gender']
            weight = request.form['weight']
            height = request.form['height']
            bloodType = request.form['bloodType']
            medicalcondition = request.form['medicalcondition']
            sleepduration = request.form['sleepduration']
            qualityofsleep = request.form['qualityofsleep']
            physicalactivitylevel = request.form['physicalactivitylevel']

            chain_data = LLMChain(llm=G_llm, prompt=prompt_temp)

            input_data = {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "bloodtype": bloodType,
                "medicalcondition": medicalcondition,
                "sleepduration": sleepduration,
                "qualityofsleep": qualityofsleep,
                "physicalactivitylevel": physicalactivitylevel,
            }

            results = chain_data.run(input_data)

            print(results)


            health_recommendations = re.findall(r'## Health Recommendations:(.*?)## Food Recommendations:', results,
                                                re.DOTALL)
            food_recommendations = re.findall(r'## Food Recommendations:(.*?)## Exercise Recommendations:', results,
                                              re.DOTALL)
            exercise_recommendations = re.findall(r'## Exercise Recommendations:(.*?)$', results, re.DOTALL)


            for category in ['health_recommendations', 'food_recommendations', 'exercise_recommendations']:
                if not locals()[category]:
                    locals()[category] = ["No " + category.replace("_", " ") + " recommendations at this time."]

            return render_template('result.html', health_recommendations=health_recommendations, food_recommendations=food_recommendations, exercise_recommendations=exercise_recommendations)  # Pass variables to template

        return render_template('index.html')

    except StopCandidateException as e:

        default_response = "Sorry, we couldn't generate a recommendation at the moment."
        return render_template('result.html', health_recommendations=[default_response],
                               food_recommendations=[default_response],
                               exercise_recommendations=[default_response])

    # except Exception as e:
    #     # Handle general exceptions
    #     print(f"An error occurred: {e}")
    #     return render_template('error.html', error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
