using UnityEngine;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json;

public class SceneCameraPoseManager : MonoBehaviour
{
    public string totalDir = @"G:/Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_1028_DLO";
    public string ScenePosePath = "G://Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_1028_DLO/ScenePose.json";

    private int lengthToRemove = 5;

    public GameObject Scene;

    void Start()
    {
        GetScenePose(ScenePosePath);
        string[] camPaths = Directory.GetFiles(totalDir + "/json");
        foreach (string camPath in camPaths)
        {
            GetCameraPose(camPath);
        }
    }

    public void GetScenePose(string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Scene_Pose = JsonToDictionary(jsonContent);
        Vector3 Scene_P = StringToVector3(Scene_Pose["Scene_P"].ToString());
        Quaternion Scene_Q = StringToQuaternion(Scene_Pose["Scene_Q"].ToString());
        Vector3 Scene_Scale = new Vector3(1.008004f, 1.008004f, 1.008004f);
        Scene.transform.position = Scene_P;
        Scene.transform.rotation = Scene_Q;
    }

    public void GetCameraPose(string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Pose_Json = JsonToDictionary(jsonContent);
        Matrix4x4 cameraToWorldMatrix = StringToMatrix(Pose_Json["cameraToWorldMatrix"].ToString());
        Matrix4x4 projectionMatrix = StringToMatrix(Pose_Json["projectionMatrix"].ToString());
        GetHololensCameraPosByMatrix(cameraToWorldMatrix, projectionMatrix);
        UnityCameraCapture(jsonPath);
    }

    public static void GetHololensCameraPosByMatrix(Matrix4x4 cameraToWorldMatrix, Matrix4x4 projectionMatrix)
    {
        // 提取相机的位置
        Vector3 cameraPosition = cameraToWorldMatrix.GetColumn(3);
        Debug.Log("cameraPosition:" + cameraPosition.ToString());

        // 提取相机的旋转
        Quaternion cameraRotation = Quaternion.LookRotation(-cameraToWorldMatrix.GetColumn(2), -cameraToWorldMatrix.GetColumn(1));
        Debug.Log("cameraRotation:" + cameraRotation.ToString());

        Camera.main.transform.position = cameraPosition;
        Camera.main.transform.rotation = cameraRotation;
        // Camera.main.projectionMatrix = projectionMatrix;
    }

    public void UnityCameraCapture(string camPath)
    {
        Camera.main.clearFlags = CameraClearFlags.SolidColor;
        Camera.main.backgroundColor = Color.white;

        RenderTexture renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
        Camera.main.targetTexture = renderTexture;
        Camera.main.Render();

        Texture2D screenshot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
        RenderTexture.active = renderTexture;

        screenshot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenshot.Apply();

        // 旋转Texture2D 180度
        Color[] pixels = screenshot.GetPixels();
        for (int y = 0; y < screenshot.height; y++)
        {
            for (int x = 0; x < screenshot.width; x++)
            {
                screenshot.SetPixel(screenshot.width - 1 - x, screenshot.height - 1 - y, pixels[x + y * screenshot.width]);
            }
        }
        screenshot.Apply();

        byte[] bytes = screenshot.EncodeToPNG();

        string Name = Path.GetFileName(camPath);
        int endIndex = Name.Length - lengthToRemove > 0 ? Name.Length - lengthToRemove : 0;
        string file = Name.Substring(0, endIndex);

        string filePath = totalDir + "/route3D_design_only/" + file + ".png";
        File.WriteAllBytes(filePath, bytes);
        RenderTexture.active = null;
        Camera.main.targetTexture = null;
        renderTexture.Release();
        Destroy(renderTexture);
        Destroy(screenshot);

        Debug.Log("Screenshot saved to: " + filePath);
    }

    private static Dictionary<string, string> JsonToDictionary(string jsonData)
    {
        return JsonConvert.DeserializeObject<Dictionary<string, string>>(jsonData);
    }

    private static Vector3 StringToVector3(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        return new Vector3(float.Parse(s[0]), float.Parse(s[1]), float.Parse(s[2]));
    }

    private static Quaternion StringToQuaternion(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        Quaternion q = new Quaternion();
        q.w = float.Parse(s[0]);
        q.x = float.Parse(s[1]);
        q.y = float.Parse(s[2]);
        q.z = float.Parse(s[3]);
        return q;
    }

    private static Matrix4x4 StringToMatrix(string str)
    {
        str = str.Replace("\n", ",").Replace("\t", ",");
        string[] s = str.Split(',');
        Matrix4x4 cameraToWorldMatix = new Matrix4x4();
        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 4; row++)
            {
                cameraToWorldMatix[row, col] = float.Parse(s[row * 4 + col]);
            }
        }
        // Debug.Log("cameraToWorldMatrix:" + cameraToWorldMatix.ToString("F6"));
        return cameraToWorldMatix;
    }
}
