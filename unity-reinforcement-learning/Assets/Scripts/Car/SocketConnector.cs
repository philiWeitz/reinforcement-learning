using System;
using System.Net;
using System.Net.Sockets;
using UnityEngine;
using System.Text;


public class SocketConnector : MonoBehaviour
{
    public int port = 11000;
    public Camera agentCamera;
    public float repeatEverySec = 0.1f;
    private byte[] _recieveBuffer = new byte[8142];

    private bool sendNextImage = true;
    private Socket clientSocket;


    void Update()
    {
        // always try to reconnect
        if (clientSocket == null || !clientSocket.Connected)
        {
            Reset();
            SetupSocketConnection();
        }
    }

    private void OnDestroy()
    {
        if (clientSocket != null)
        {
            clientSocket.Dispose();
            clientSocket = null;
        }
    }

    private void Reset()
    {
        CancelInvoke();
        sendNextImage = true;
    }

    private void SetupSocketConnection()
    {
        try
        {
            clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            clientSocket.Connect(new IPEndPoint(IPAddress.Loopback, port));
            clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);

            InvokeRepeating("SendData", 1, repeatEverySec);
            Debug.Log("Socket connection established");
        }
        catch (SocketException) {
            clientSocket.Dispose();
            clientSocket = null;
        }
    }

    private void ReceiveCallback(IAsyncResult AR)
    {
        int recieved = clientSocket.EndReceive(AR);

        if (recieved > 0)
        {
            byte[] data = new byte[recieved];
            Buffer.BlockCopy(_recieveBuffer, 0, data, 0, recieved);

            string dataString = Encoding.UTF8.GetString(data);
            if (dataString == "RESET")
            {
                Environment.instance.resetEnvironment = true;
            }
            else
            {
                MoveModel moveModel = MoveModel.FromJsonBytes(data);
                Environment.instance.networkMoveModel = moveModel;
            }

            sendNextImage = true;
        }

        // start receiving data again
        clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);
    }

    private void SendData()
    {
        // only send picture when the simmulation is running
        if (clientSocket.Connected && sendNextImage == true)
        {
            Texture2D agentTexture = GetAgentCameraTexture();

            if (agentTexture)
            {
                Color[] colors = agentTexture.GetPixels();
                FrameModel frame = new FrameModel(Environment.instance.isOnTrack,
                    Environment.instance.isTerminalState, Environment.instance.isFinishReached, colors);

                byte[] data = frame.ToJsonBytes();
                Destroy(agentTexture);

                SocketAsyncEventArgs socketAsyncData = new SocketAsyncEventArgs();
                socketAsyncData.SetBuffer(data, 0, data.Length);
                clientSocket.SendAsync(socketAsyncData);

                // wait until the data is read
                sendNextImage = false;
            }
        }
        else if (clientSocket.Connected && clientSocket.Poll(100, SelectMode.SelectRead)) {
            Debug.Log("Socket connection was closed unexpectedly. Reseting socket...");
            clientSocket.Dispose();
            clientSocket = null;
            Environment.instance.resetEnvironment = true;
        }
    }

    private Texture2D GetAgentCameraTexture()
    {
        Rect rect = new Rect(0, 0, 120, 50);
        RenderTexture renderTexture = new RenderTexture(120, 50, 24);
        Texture2D screenShot = new Texture2D(120, 50, TextureFormat.RGB24, false);

        agentCamera.targetTexture = renderTexture;
        agentCamera.Render();

        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(rect, 0, 0);

        agentCamera.targetTexture = null;
        RenderTexture.active = null;

        Destroy(renderTexture);
        renderTexture = null;

        return screenShot;
    }
}
