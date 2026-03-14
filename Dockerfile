ARG BUILD_FROM
FROM $BUILD_FROM

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libusb-1.0-0 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY hand_tracker.py .
COPY run.sh .
# Konfiguration erfolgt jetzt über das Dashboard-UI
RUN sed -i 's/\r//' /app/run.sh && chmod a+x /app/run.sh

RUN mkdir -p /etc/services.d/hand_control_pro \
    && printf '#!/usr/bin/with-contenv bashio\nexec /app/run.sh\n' \
    > /etc/services.d/hand_control_pro/run \
    && sed -i 's/\r//' /etc/services.d/hand_control_pro/run \
    && chmod a+x /etc/services.d/hand_control_pro/run

# Explizit s6 als Entrypoint setzen
ENTRYPOINT ["/init"]