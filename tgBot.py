import torch
from torchvision import models, transforms
from PIL import Image
import io
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext, Application

MODEL_PATH = 'model_resnet50_pretrained.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50()
fc_layer = model.fc
model.fc = torch.nn.Linear(fc_layer.in_features, 10)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_image(image: Image.Image) -> str:
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    class_names = ['Chameleon', 'Crocodile_Alligator', 'Frog', 'Gecko', 'Iguana', 'Lizard', 'Salamander', 'Snake', 'Toad', 'Turtle_Tortoise']
    return f"Предсказанный класс: {class_names[predicted.item()]}"

async def handle_photo(update: Update, context: CallbackContext):
    photo = await update.message.photo[-1].get_file()
    photo_bytes = io.BytesIO(await photo.download_as_bytearray())
    image = Image.open(photo_bytes)

    prediction = analyze_image(image)
    
    await update.message.reply_text(prediction)

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Отправьте изображение, и я проанализирую его с помощью модели!")

def main():
    # Вставьте свой токен
    TOKEN = '1596944937:AAHDpnFZmLolunh_a11b5THEOrQAFZDbcys'
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == '__main__':
    main()
