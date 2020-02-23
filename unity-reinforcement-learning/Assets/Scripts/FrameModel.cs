using System;
using System.Text;
using UnityEngine;

[Serializable]
public class FrameModel
{
    public bool isOnTrack;

    public uint[] colors;

    public FrameModel(bool isOnTrack, Color[] colors)
    {
        this.isOnTrack = isOnTrack;

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
