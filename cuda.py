import requests

api_key = "ca60df8b71901a3837bd3db7e13bc006"
city = "Kigali"
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

response = requests.get(url)
print(response.status_code)
print(response.text)
