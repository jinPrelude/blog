# 1. 설치
설치 방법은 아래와 같다:
## 1. chruby & ruby-install 설치
### 1. chruby란?
맥에는 기본 ruby가 설치되어 있다. 하지만 2버전임.
jekyll을 돌리기 위해서는 더 높은 ruby 버전이 필요함.
여러가지의 ruby 버전을 관리할 수 있도록 도와주는 라이브러리가 chruby.
chruby를 깔고, ruby-install를 사용하여 chruby에 우리가 원하는 ruby3를 설치할 것임.

### 2. `chruby`와 `ruby-install` 설치
```bash
brew install chruby ruby-install xz
```

### 3. ruby 3.1.3 설치
```bash
ruby-install ruby 3.1.3
```

### 4. `chruby` 자동 실행 설정
```bash
echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
echo "chruby ruby-3.1.3" >> ~/.zshrc # run 'chruby' to see actual version
```

### 5. 확인
터미널 다시 시작해서 아래와 같이 쳤을 때 3.1.3 잘 설치되어 있는지 확인:
```bash
ruby -v
```

### jekyll 설치
```bash
gem install jekyll
```