# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# import torch
# import re 
# from fastapi.templating import Jinja2Templates # UI
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles

# app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
# tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# # device
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

# model.to(device)

# templates = Jinja2Templates(directory=".")

# class DialogueInput(BaseModel):
#     dialogue: str

# def clean_data(text):
#     text = re.sub(r"\r\n|\n|\r", " ", text)
#     text = re.sub(r"<.*?>", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     text = text.strip().lower()
#     return text

# def summarize_dialogue(dialogue : str) -> str:
#     dialogue = clean_data(dialogue) # clean

#     # tokenize
#     inputs = tokenizer(
#         dialogue,
#         padding="max_length",
#         max_length=512,
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)

#     # generate the summary => token ids
#     model.to(device)
#     targets = model.generate(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_length=150,
#         num_beams=4,
#         early_stopping=True
#     )
    
#     # decoded our output
#     summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
#     return summary


# # API endpoints
# @app.post("/summarize/")
# async def summarize(dialogue_input: DialogueInput):
#     summary = summarize_dialogue(dialogue_input.dialogue)
#     return {"summary": summary}

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})



from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import torch
import re
import logging


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI
app = FastAPI(
    title="SnapSummary",
    description="AI-powered Text Summarization using T5",
    version="1.0"
)


# Device configuration
device = torch.device("cpu")

# Limit CPU threads to reduce resource usage
torch.set_num_threads(1)


# Model path
MODEL_PATH = "./Artifacts"


# Load tokenizer
logger.info("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


# Load model
logger.info("Loading summarization model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

logger.info("Model loaded successfully!")


# HTML templates
templates = Jinja2Templates(directory=".")


# Input validation
class DialogueInput(BaseModel):
    dialogue: str = Field(min_length=1)


# Text cleaning function
def clean_data(text: str) -> str:

    # Remove line breaks
    text = re.sub(r"\r\n|\n|\r", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


# Summarization function
def summarize_dialogue(dialogue: str) -> str:

    # Clean input text
    dialogue = clean_data(dialogue)

    if not dialogue:
        raise ValueError("Please enter some text.")

    # Tokenize input
    inputs = tokenizer(
        dialogue,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move input to CPU
    inputs = inputs.to(device)

    # Generate summary
    with torch.inference_mode():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            num_beams=1,
            do_sample=False
        )

    # Decode generated tokens
    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# Summarization API
@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):

    try:

        summary = summarize_dialogue(
            dialogue_input.dialogue
        )

        return {
            "summary": summary
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        logger.exception("Summarization failed")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the summary."
        )


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# Health check
@app.get("/health")
def health_check():

    return {
        "status": "Server is running",
        "model_loaded": model is not None
    }