FROM nvidia/cuda:11.1.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y --ignore-missing --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    htop \
    git \
    locales \
    net-tools \
    openssh-server \
    ruby \
    ruby-colorize \
    ruby-dev \
    tmux \
    unzip \
    vim \
    ssh \ 
    sudo \ 
    wget \
    unzip \
    x11-apps\
    xauth \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    zsh && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN gem install public_suffix -v 4.0.7
RUN gem install colorls -v 1.4.4

# locale setting (set to ko_KR_UTF8)
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8 && localedef -f UTF-8 -i en_US en_US.UTF-8

# zsh settings
RUN chsh -s `which zsh`
RUN curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
RUN zsh -c setopt correct
RUN git clone https://github.com/djui/alias-tips.git /root/.oh-my-zsh/plugins/alias-tips
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /root/.oh-my-zsh/plugins/zsh-syntax-highlighting
RUN git clone https://github.com/zsh-users/zsh-autosuggestions /root/.oh-my-zsh/plugins/zsh-autosuggestions
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git /root/.oh-my-zsh/themes/powerlevel10k
# mkdir ssh dir
RUN mkdir /var/run/sshd

# Copy dot files that pip, zsh, vim, tmux settings
RUN curl -L -O https://github.com/kairos03/Ocean-public/raw/main/scripts/dot_files.zip
RUN unzip -o dot_files.zip -d /root/
RUN rm -f dot_files.zip

# replace sshd_config
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN sed -ri 's/^#?AllowTcpForwarding\s+.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?X11DisplayOffset\s+.*/X11DisplayOffset 10/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?PrintLastLog\s+.*/PrintLastLog yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?TCPKeepAlive\s+.*/TCPKeepAlive yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?#ClientAliveInterval\s+.*/#ClientAliveInterval 300/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?X11UseLocalhost\s+.*/X11UseLocalhost no/' /etc/ssh/sshd_config
# make .ssh
RUN mkdir /root/.ssh


RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm /tmp/miniconda.sh
RUN conda update -y conda

# Create virtual env
ENV PYTHON_VERSION=3.8
ENV CONDA_ENV_NAME=test

RUN conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION
ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# ssh initialization
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config
    
RUN /bin/bash -c "source activate ${CONDA_ENV_NAME} && \
    python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html \
    numpy==1.23.5  \
    tqdm \
    transformers==4.15.0 \
    Cython \
    ninja \
    "

COPY . .

# Set environment variables to enable GPU access within the container
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# # Expose any necessary ports (e.g., for API services)
# EXPOSE 8080

# Start your application (modify this according to your application)
CMD [ "/bin/bash" ]