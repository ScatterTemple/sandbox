from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class Sample(BaseModel):
    a: int
    b: int


sample = Sample(a=0, b=0)


@app.post("/set_values/")
def set_values(values: Sample):
    global sample
    sample.a = values.a
    sample.b = values.b
    return {"message": "Values set successfully"}


@app.get("/calculate_sum/")
def calculate_sum():
    total = sample.a + sample.b
    return {"sum": total}


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)
