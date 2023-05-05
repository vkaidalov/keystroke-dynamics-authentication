
# Serves the app at localhost:8866 by default.
voila app.py \
    --Voila.log_level=20 \
    --VoilaConfiguration.show_tracebacks=True \
    --autoreload=True \
    --no-browser \
    --enable_nbextensions=True \
    --VoilaConfiguration.extension_language_mapping='{".py": "python"}'
