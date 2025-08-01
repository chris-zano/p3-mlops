#!/bin/bash
set -xe

REGION="us-east-1"
ECR_IMAGE="${ecr_image}"
MFLOW_SERVER_URL="${mflow_server_ip}"
echo 'export MFLOW_SERVER_URL="${mflow_server_ip}"' >> ~/.bashrc

# Install dependencies
apt-get update -y
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    unzip

# Install Docker (official method for Ubuntu 24+)
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker
systemctl enable docker
systemctl start docker

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws/

# Add docker to the user group
usermod -aG docker ubuntu
newgrp docker


# === Add trainservice.sh ===
cat <<'EOF' > /usr/local/bin/trainservice.sh
#!/bin/bash
set -euo pipefail

DOCKER_IMAGE="${ecr_image}"
AWS_REGION="eu-west-1"
CONTAINER_NAME="train_model"
export MFLOW_SERVER_URL="${mflow_server_ip}"

echo "[INFO] Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin $(echo "$DOCKER_IMAGE" | awk -F/ '{print $1}')

echo "[INFO] Pulling Docker image: $DOCKER_IMAGE"
docker pull "$DOCKER_IMAGE"

echo "[INFO] Running Docker container: $CONTAINER_NAME"
if docker run --rm \
  --name "$CONTAINER_NAME" \
  -e MFLOW_SERVER_URL="$MFLOW_SERVER_URL" \
  "$DOCKER_IMAGE"; then
    echo "[INFO] Training completed successfully."
else
    echo "[ERROR] Training container failed. Not stopping instance."
    exit 1
fi

echo "[INFO] Cleaning up old containers/images"
docker container prune -f
docker image rm "$DOCKER_IMAGE" -f || true

echo "[INFO] Fetching EC2 instance ID..."
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)

if [ -z "$INSTANCE_ID" ]; then
    echo "[ERROR] Could not retrieve instance ID. Aborting shutdown."
    exit 1
fi

echo "[INFO] Stopping instance $INSTANCE_ID in $AWS_REGION"
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"
EOF

chmod +x /usr/local/bin/trainservice.sh

# === Add systemd unit ===
cat <<'EOF' > /etc/systemd/system/train-docker-job.service
[Unit]
Description=Run ECR training job and stop instance
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/trainservice.sh
Environment="ECR_IMAGE=$${ECR_IMAGE}"
Environment="MFLOW_SERVER_URL=$${MFLOW_SERVER_URL}"
Environment="REGION=$${REGION}"
RemainAfterExit=no
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd & enable the service
systemctl daemon-reload
systemctl enable train-docker-job.service
systemctl start train-docker-job.service