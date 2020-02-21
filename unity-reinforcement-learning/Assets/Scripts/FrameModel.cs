using System;
using System.Text;
using UnityEngine;

[Serializable]
public class FrameModel
{
    public string data;

    public uint[] colors;

    public FrameModel(string data, Color[] colors)
    {
        this.data = data;

        this.colors = new uint[colors.Length];

        for (int i = 0; i < colors.Length; ++i)
        {
            this.colors[i] = (uint)Math.Floor(255 * colors[i].grayscale);
        }
    }

    public byte[] ToJsonBytes()
    {
        return Encoding.UTF8.GetBytes(JsonUtility.ToJson(this));
    }
}
