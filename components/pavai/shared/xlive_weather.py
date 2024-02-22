# import the module
import python_weather

import asyncio
import os

def getweather(location:str,unit=python_weather.METRIC):
  # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
  client=python_weather.Client(unit=python_weather.METRIC)
  weather=client.get(location)
  return weather, python_weather.METRIC

if __name__ == '__main__':
  # see https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
  # for more details
  if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  result, unit= getweather(location="Toronto",unit=python_weather.METRIC)
  print(result, unit.temperature)
