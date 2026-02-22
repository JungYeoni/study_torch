{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPSnzk4fqSO2/vWNnUdfthm",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JungYeoni/study_torch/blob/main/nn_Module%EA%B3%BC_%ED%81%B4%EB%9E%98%EC%8A%A4%EB%A1%9C_%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#  nn.Module과 클래스로 구현하기"
      ],
      "metadata": {
        "id": "iC_NOvL7UtV9"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "학습자료\n",
        "> https://wikidocs.net/55409"
      ],
      "metadata": {
        "id": "YkGY8TXcVZZD"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "- 파이토치에 이미 구현되어져 제공되고 있는 함수들을 불러와 선형 회귀 모델 구현"
      ],
      "metadata": {
        "id": "RvdBMOSEVXpt"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 단순 선형 회귀 구현하기"
      ],
      "metadata": {
        "id": "a5Gkq9r8U5wu"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "2MQxeiBdUqUM"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "torch.manual_seed(1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YxMttKaVhc3D",
        "outputId": "64a27aac-257d-46f8-d623-2203ba539fb7"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<torch._C.Generator at 0x7a86a8a1fd10>"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 데이터 선언\n",
        "x_train = torch.FloatTensor([[1], [2], [3]])\n",
        "y_train = torch.FloatTensor([[2], [4], [6]])"
      ],
      "metadata": {
        "id": "-UONrmtHhpEh"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "`nn.Linear(input_dim, output_dim)`: 입력 차원과 출력 차원을 인자로 받는다"
      ],
      "metadata": {
        "id": "GH4MveX3h7K5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 모델 선언 및 초기화\n",
        "model = nn.Linear(1, 1)"
      ],
      "metadata": {
        "id": "80wvpOs_h1O7"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# model에는 가중치와 편향이 저장되어 있음\n",
        "print(list(model.parameters()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7-cnyAmUh4_i",
        "outputId": "83669418-3b97-4c26-e26c-a788ad43249d"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[Parameter containing:\n",
            "tensor([[0.5153]], requires_grad=True), Parameter containing:\n",
            "tensor([-0.4414], requires_grad=True)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# optimizer 설정 - SGD를 사용\n",
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)"
      ],
      "metadata": {
        "id": "qEP9lImSiK8V"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 전체 훈련 데이터에 대해 경사하강법을 2000회 반복\n",
        "nb_epochs = 2000\n",
        "\n",
        "for epoch in range(nb_epochs + 1):\n",
        "  # H(x) 계산\n",
        "  prediction = model(x_train)\n",
        "\n",
        "  # cost 계산\n",
        "  cost = F.mse_loss(prediction, y_train) # 파이토치에서 제공하는 평균제곱오차 사용\n",
        "\n",
        "  '''cost로 H(x) 계산하는 부분'''\n",
        "\n",
        "  # gradient를 0으로 초기화\n",
        "  optimizer.zero_grad()\n",
        "\n",
        "  # 비용 함수를 미분하여 gradient 계산\n",
        "  cost.backward()\n",
        "\n",
        "  # 파라미터 업데이트\n",
        "  optimizer.step()\n",
        "\n",
        "  if epoch % 100 == 0:\n",
        "    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hz0y5qwXiaNj",
        "outputId": "b9b3e9db-432a-4009-bfb1-6f100875aca0"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch    0/2000 Cost: 13.103541\n",
            "Epoch  100/2000 Cost: 0.002791\n",
            "Epoch  200/2000 Cost: 0.001724\n",
            "Epoch  300/2000 Cost: 0.001066\n",
            "Epoch  400/2000 Cost: 0.000658\n",
            "Epoch  500/2000 Cost: 0.000407\n",
            "Epoch  600/2000 Cost: 0.000251\n",
            "Epoch  700/2000 Cost: 0.000155\n",
            "Epoch  800/2000 Cost: 0.000096\n",
            "Epoch  900/2000 Cost: 0.000059\n",
            "Epoch 1000/2000 Cost: 0.000037\n",
            "Epoch 1100/2000 Cost: 0.000023\n",
            "Epoch 1200/2000 Cost: 0.000014\n",
            "Epoch 1300/2000 Cost: 0.000009\n",
            "Epoch 1400/2000 Cost: 0.000005\n",
            "Epoch 1500/2000 Cost: 0.000003\n",
            "Epoch 1600/2000 Cost: 0.000002\n",
            "Epoch 1700/2000 Cost: 0.000001\n",
            "Epoch 1800/2000 Cost: 0.000001\n",
            "Epoch 1900/2000 Cost: 0.000000\n",
            "Epoch 2000/2000 Cost: 0.000000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#  임의의 값을 넣어 모델이 예측하는 값 확인\n",
        "new_var = torch.FloatTensor([[4.0]])\n",
        "\n",
        "# 입력한 값 4에 대해서 예측 값 y를 리턴받아 pred_y에 저장\n",
        "pred_y = model(new_var)\n",
        "\n",
        "print('훈련 후 입력이 4일 때의 예측값:', pred_y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "onCpZCT8kjsf",
        "outputId": "2228bc8e-96e3-4cb1-a279-d73e40068466"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "훈련 후 입력이 4일 때의 예측값: tensor([[7.9989]], grad_fn=<AddmmBackward0>)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 학습 후의 파라미터 출력\n",
        "print(list(model.parameters()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_mLaWEx4k72D",
        "outputId": "c340ac29-ec21-49df-a4db-7c0fd78447ab"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[Parameter containing:\n",
            "tensor([[1.9994]], requires_grad=True), Parameter containing:\n",
            "tensor([0.0014], requires_grad=True)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "- forward 연산: 가설을 통해 입력 $x$로부터 예측된 $y$를 얻는 것\n",
        "- backward 연산: 학습 과정에서 비용 함수를 미분하여 기울기를 구하는 것"
      ],
      "metadata": {
        "id": "Nfwg2D4-k6rW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 다중 선형 회귀 구현하기"
      ],
      "metadata": {
        "id": "pMcoXkd1igA1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F"
      ],
      "metadata": {
        "id": "iATdu-ToiiN9"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "torch.manual_seed(1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a9WSlsLgmKO7",
        "outputId": "0f7bf55e-39e9-46ca-abde-62f84b63ef54"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<torch._C.Generator at 0x7a86a8a1fd10>"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 데이터 선언 - 3개의 x로부터 하나의 y를 예측하는 문제\n",
        "x_train = torch.FloatTensor([[73, 80, 75],\n",
        "                             [93, 88, 93],\n",
        "                             [89, 91, 90],\n",
        "                             [96, 98, 100],\n",
        "                             [73, 66, 70]])\n",
        "y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])"
      ],
      "metadata": {
        "id": "ZybIPoAzmLNl"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 모델 선언 및 초기화\n",
        "model = nn.Linear(3, 1)"
      ],
      "metadata": {
        "id": "J2JAQaFfmRDW"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(list(model.parameters()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "86uFI8ulnA1W",
        "outputId": "248d22fa-188f-41f3-ccdd-8cf90928498a"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[Parameter containing:\n",
            "tensor([[ 0.2975, -0.2548, -0.1119]], requires_grad=True), Parameter containing:\n",
            "tensor([0.2710], requires_grad=True)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# optimizer 정의\n",
        "optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)"
      ],
      "metadata": {
        "id": "tZc8q9cNnDvc"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 전체 훈련 데이터에 대해 경사하강법을 2000회 반복\n",
        "nb_epochs = 2000\n",
        "\n",
        "for epoch in range(nb_epochs + 1):\n",
        "  # H(x) 계산\n",
        "  prediction = model(x_train)\n",
        "\n",
        "  # cost 계산\n",
        "  cost = F.mse_loss(prediction, y_train) # 파이토치에서 제공하는 평균제곱오차 사용\n",
        "\n",
        "  '''cost로 H(x) 계산하는 부분'''\n",
        "\n",
        "  # gradient를 0으로 초기화\n",
        "  optimizer.zero_grad()\n",
        "\n",
        "  # 비용 함수를 미분하여 gradient 계산\n",
        "  cost.backward()\n",
        "\n",
        "  # 파라미터 업데이트\n",
        "  optimizer.step()\n",
        "\n",
        "  if epoch % 100 == 0:\n",
        "    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3OlkFGnhnP_l",
        "outputId": "69c906c6-3c3d-4548-8afb-3e4f2b0659d4"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch    0/2000 Cost: 31667.597656\n",
            "Epoch  100/2000 Cost: 0.225993\n",
            "Epoch  200/2000 Cost: 0.223911\n",
            "Epoch  300/2000 Cost: 0.221941\n",
            "Epoch  400/2000 Cost: 0.220059\n",
            "Epoch  500/2000 Cost: 0.218271\n",
            "Epoch  600/2000 Cost: 0.216575\n",
            "Epoch  700/2000 Cost: 0.214950\n",
            "Epoch  800/2000 Cost: 0.213413\n",
            "Epoch  900/2000 Cost: 0.211952\n",
            "Epoch 1000/2000 Cost: 0.210560\n",
            "Epoch 1100/2000 Cost: 0.209232\n",
            "Epoch 1200/2000 Cost: 0.207967\n",
            "Epoch 1300/2000 Cost: 0.206761\n",
            "Epoch 1400/2000 Cost: 0.205619\n",
            "Epoch 1500/2000 Cost: 0.204522\n",
            "Epoch 1600/2000 Cost: 0.203484\n",
            "Epoch 1700/2000 Cost: 0.202485\n",
            "Epoch 1800/2000 Cost: 0.201542\n",
            "Epoch 1900/2000 Cost: 0.200635\n",
            "Epoch 2000/2000 Cost: 0.199769\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 임의의 입력을 통해 예측값 확인\n",
        "new_var = torch.FloatTensor([[73, 80, 75]])\n",
        "pred_y = model(new_var)\n",
        "print('훈련 후 입력이 73, 80, 75 일 때의 예측값: ', pred_y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aha9e2KGplx8",
        "outputId": "687d3024-4bc4-40c2-959b-13e608432604"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "훈련 후 입력이 73, 80, 75 일 때의 예측값:  tensor([[151.2305]], grad_fn=<AddmmBackward0>)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 파라미터 확인\n",
        "print(list(model.parameters()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AikNGsMqp3_D",
        "outputId": "e93d8f01-3206-4a6e-e35b-dbb2343e324c"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[Parameter containing:\n",
            "tensor([[0.9778, 0.4539, 0.5768]], requires_grad=True), Parameter containing:\n",
            "tensor([0.2802], requires_grad=True)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 모델을 클래스로 구현하기"
      ],
      "metadata": {
        "id": "8spiGYivsbAF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 단순선형회귀"
      ],
      "metadata": {
        "id": "4nPvfdflySFI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F"
      ],
      "metadata": {
        "id": "T8eL2y-Myzac"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "torch.manual_seed(1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9rYw8za0y5dV",
        "outputId": "11083c68-5401-48b1-d202-2cdfe837413c"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<torch._C.Generator at 0x7a86a8a1fd10>"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_train = torch.FloatTensor([[1], [2], [3]])\n",
        "y_train = torch.FloatTensor([[2], [4], [6]])"
      ],
      "metadata": {
        "id": "m70-k1hLyvAF"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class LinearRegressionModel(nn.Module):\n",
        "  def __init__(self):\n",
        "    super().__init__() # nn.Module 클래스의 속성들을 가지고 초기화됨\n",
        "    self.linear = nn.Linear(1, 1) # 단순 선형회귀이므로\n",
        "\n",
        "  def forward(self, x):\n",
        "    return self.linear(x)"
      ],
      "metadata": {
        "id": "7-mvBCDosVyg"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = LinearRegressionModel()"
      ],
      "metadata": {
        "id": "G9KhA-UPx9RC"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)"
      ],
      "metadata": {
        "id": "5e8xGOuoyidy"
      },
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 전체 훈련 데이터에 대해 경사하강법을 2000회 반복\n",
        "nb_epochs = 2000\n",
        "\n",
        "for epoch in range(nb_epochs + 1):\n",
        "  # H(x) 계산\n",
        "  prediction = model(x_train)\n",
        "\n",
        "  # cost 계산\n",
        "  cost = F.mse_loss(prediction, y_train) # 파이토치에서 제공하는 평균제곱오차 사용\n",
        "\n",
        "  '''cost로 H(x) 계산하는 부분'''\n",
        "\n",
        "  # gradient를 0으로 초기화\n",
        "  optimizer.zero_grad()\n",
        "\n",
        "  # 비용 함수를 미분하여 gradient 계산\n",
        "  cost.backward()\n",
        "\n",
        "  # 파라미터 업데이트\n",
        "  optimizer.step()\n",
        "\n",
        "  if epoch % 100 == 0:\n",
        "    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nr8n0kRIyqyS",
        "outputId": "3b049b54-cb98-4027-da19-e4e059f158d3"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch    0/2000 Cost: 13.103541\n",
            "Epoch  100/2000 Cost: 0.002791\n",
            "Epoch  200/2000 Cost: 0.001724\n",
            "Epoch  300/2000 Cost: 0.001066\n",
            "Epoch  400/2000 Cost: 0.000658\n",
            "Epoch  500/2000 Cost: 0.000407\n",
            "Epoch  600/2000 Cost: 0.000251\n",
            "Epoch  700/2000 Cost: 0.000155\n",
            "Epoch  800/2000 Cost: 0.000096\n",
            "Epoch  900/2000 Cost: 0.000059\n",
            "Epoch 1000/2000 Cost: 0.000037\n",
            "Epoch 1100/2000 Cost: 0.000023\n",
            "Epoch 1200/2000 Cost: 0.000014\n",
            "Epoch 1300/2000 Cost: 0.000009\n",
            "Epoch 1400/2000 Cost: 0.000005\n",
            "Epoch 1500/2000 Cost: 0.000003\n",
            "Epoch 1600/2000 Cost: 0.000002\n",
            "Epoch 1700/2000 Cost: 0.000001\n",
            "Epoch 1800/2000 Cost: 0.000001\n",
            "Epoch 1900/2000 Cost: 0.000000\n",
            "Epoch 2000/2000 Cost: 0.000000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 다중 선형 회귀"
      ],
      "metadata": {
        "id": "6JNMpfUNyT8g"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F"
      ],
      "metadata": {
        "id": "59ixN_6JyBlW"
      },
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "torch.manual_seed(1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I6zbyAi3zDIn",
        "outputId": "3cb32783-27cb-441f-db22-3c24475fe6bf"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<torch._C.Generator at 0x7a86a8a1fd10>"
            ]
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 데이터\n",
        "x_train = torch.FloatTensor([[73, 80, 75],\n",
        "                             [93, 88, 93],\n",
        "                             [89, 91, 90],\n",
        "                             [96, 98, 100],\n",
        "                             [73, 66, 70]])\n",
        "y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])"
      ],
      "metadata": {
        "id": "BKnfcgfXzGNo"
      },
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class MultivariateLinearRegression(nn.Module):\n",
        "  def __init__(self):\n",
        "    super().__init__()\n",
        "    self.linear = nn.Linear(3, 1)\n",
        "\n",
        "  def forward(self, x):\n",
        "    return self.linear(x)"
      ],
      "metadata": {
        "id": "CCYw26OvzI6w"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = MultivariateLinearRegression()"
      ],
      "metadata": {
        "id": "45fVhGCLzm9l"
      },
      "execution_count": 44,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 전체 훈련 데이터에 대해 경사하강법을 2000회 반복\n",
        "nb_epochs = 2000\n",
        "\n",
        "for epoch in range(nb_epochs + 1):\n",
        "  # H(x) 계산\n",
        "  prediction = model(x_train)\n",
        "\n",
        "  # cost 계산\n",
        "  cost = F.mse_loss(prediction, y_train) # 파이토치에서 제공하는 평균제곱오차 사용\n",
        "\n",
        "  '''cost로 H(x) 계산하는 부분'''\n",
        "\n",
        "  # gradient를 0으로 초기화\n",
        "  optimizer.zero_grad()\n",
        "\n",
        "  # 비용 함수를 미분하여 gradient 계산\n",
        "  cost.backward()\n",
        "\n",
        "  # 파라미터 업데이트\n",
        "  optimizer.step()\n",
        "\n",
        "  if epoch % 100 == 0:\n",
        "    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d5Nwxhvlzobm",
        "outputId": "35cecc62-4697-45be-ff9f-9ac2d1af5567"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch    0/2000 Cost: 31667.597656\n",
            "Epoch  100/2000 Cost: 31667.597656\n",
            "Epoch  200/2000 Cost: 31667.597656\n",
            "Epoch  300/2000 Cost: 31667.597656\n",
            "Epoch  400/2000 Cost: 31667.597656\n",
            "Epoch  500/2000 Cost: 31667.597656\n",
            "Epoch  600/2000 Cost: 31667.597656\n",
            "Epoch  700/2000 Cost: 31667.597656\n",
            "Epoch  800/2000 Cost: 31667.597656\n",
            "Epoch  900/2000 Cost: 31667.597656\n",
            "Epoch 1000/2000 Cost: 31667.597656\n",
            "Epoch 1100/2000 Cost: 31667.597656\n",
            "Epoch 1200/2000 Cost: 31667.597656\n",
            "Epoch 1300/2000 Cost: 31667.597656\n",
            "Epoch 1400/2000 Cost: 31667.597656\n",
            "Epoch 1500/2000 Cost: 31667.597656\n",
            "Epoch 1600/2000 Cost: 31667.597656\n",
            "Epoch 1700/2000 Cost: 31667.597656\n",
            "Epoch 1800/2000 Cost: 31667.597656\n",
            "Epoch 1900/2000 Cost: 31667.597656\n",
            "Epoch 2000/2000 Cost: 31667.597656\n"
          ]
        }
      ]
    }
  ]
}
