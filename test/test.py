
def get_user_input():

    time_of_day = input("Enter time of day (Morning, Afternoon, Evening, Night): ").lower()
   
    try:
        steps = float(input("Enter number of steps: "))
        temperature = float(input("Enter temperature: "))
        wind_speed = float(input("Enter wind speed: "))
    except ValueError:
        print("Error: Please enter valid numeric values for steps, temperature, and wind speed.")
        return None
    
    genre_mood = input("Enter genre/mood (Chill, Upbeat, etc.): ").lower()


    if time_of_day not in [label.lower() for label in le_time.classes_]:
        print(f"Error: Unknown time_of_day '{time_of_day}'. Please enter a valid time of day.")
        print("Valid time of day options are:", le_time.classes_)
        return None
    

    if genre_mood not in [label.lower() for label in le_genre.classes_]:
        print(f"Error: Unknown genre/mood '{genre_mood}'. Please enter a valid genre/mood.")
        print("Valid genre/mood options are:", le_genre.classes_)
        return None


    time_of_day = le_time.transform([time_of_day])[0]
    genre_mood = le_genre.transform([genre_mood])[0]

    return [time_of_day, steps, temperature, wind_speed, genre_mood]


def predict_music(model, user_input):
    model.eval()  
    user_input_tensor = torch.tensor([user_input], dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(user_input_tensor)
        predicted_class = torch.argmax(y_pred, dim=1).item() 
        predicted_music = le_music.inverse_transform([predicted_class])[0]  
    return predicted_music


def main():
    user_input = get_user_input()
    if user_input is not None:
        predicted_music = predict_music(model, user_input)
        print(f"Recommended music: {predicted_music}")


if __name__ == "__main__":
    main()
