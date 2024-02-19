#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

using Onnx;

#endregion

namespace Ceres.Base.Misc.ONNX
{
  /// <summary>
  /// Represents an ONNX model and provides some helper functions for working with ONNX models.
  /// </summary>
  public class ONNXNet
  {
    /// <summary>
    /// Underlying ONNX protobuf.
    /// </summary>
    public readonly ModelProto Model;


    private string savedFileName;


    /// <summary>
    /// Constructor (from file name).
    /// </summary>
    /// <param name="onnxFileName"></param>
    public ONNXNet(string onnxFileName)
    {
      savedFileName = onnxFileName;
      Model = ModelProto.Parser.ParseFromFile(onnxFileName);
    }


    /// <summary>
    /// Constructor (from an existing ModelProto).
    /// </summary>
    /// <param name="model"></param>
    public ONNXNet(ModelProto model) => Model = model;

    /// <summary>
    /// Returns the number of parameters in the model.
    /// </summary>
    public long NumParams => ONNXHelpers.NumParameters(Model);


    /// <summary>
    /// Name of the ONNX file (writes out to temporary file if not already sourced from a file).
    /// </summary>
    public string ONNXFileName
    {
      get
      {
        if (savedFileName == null)
        {
          savedFileName = Path.GetTempFileName();
          Model.WriteToFile(savedFileName);
        }

        return savedFileName;
      }
    }




    /// <summary>
    /// Returns initializer for a Parameter node with a specified name. 
    /// </summary>
    /// <param name="baseNodeName"></param>
    /// <returns></returns>
    public ONNXTensorProto InitializerForNode(string baseNodeName) => ParametersForNode(baseNodeName).Find(t => t.Name == baseNodeName);

    /// <summary>
    /// Returns weights for a node with a specified name. 
    /// </summary>
    /// <param name="baseNodeName"></param>
    /// <returns></returns>
    public ONNXTensorProto InitializerWeightsForNode(string baseNodeName) => ParametersForNode(baseNodeName).Find(t => t.Name.EndsWith(".weight"));

    /// <summary>
    /// Returns biases for a node with a specified name.
    /// </summary>
    /// <param name="baseNodeName"></param>
    /// <returns></returns>
    public ONNXTensorProto InitializerBiasesForNode(string baseNodeName) => ParametersForNode(baseNodeName).Find(t => t.Name.EndsWith(".bias"));


    /// <summary>
    /// Returns list of all parameters for a node with a specified name. 
    /// </summary>
    /// <param name="baseNodeName"></param>
    /// <returns></returns>
    public List<ONNXTensorProto> ParametersForNode(string baseNodeName)
    {
      IEnumerable<TensorProto> thisNodeInitializer = Model.Graph.Initializer.Where(s => s.Name.StartsWith(baseNodeName));

      List<ONNXTensorProto> ret = new(thisNodeInitializer.Count());
      foreach (TensorProto init in thisNodeInitializer)
      {
        ret.Add(new ONNXTensorProto(init));
      }
      return ret;
    }


    public NodeProto? NodeWithName(string nodeName) => Model.Graph.Node.FirstOrDefault(graph => graph.Name == nodeName);


    /// <summary>
    /// Returns a new model with a specified new output (representing output of one of the inner nodes) added.
    /// </summary>
    /// <param name="nodeName"></param>
    /// <param name="outerDim"></param>
    /// <returns></returns>
    public T WithAddedOutputNode<T>(string nodeName, int outerDim, Func<ModelProto, T> converter)
    {

      NodeProto node = NodeWithName(nodeName);
      if (node == null)
      {
        throw new Exception("Node not found: " + nodeName);
      }

      if (node.OpType != "MatMul")
      {
        throw new Exception("Implementation limitation: only MatMul nodes are supported (able to infer output shape).");
      }

      string weightsInitializerName = node.Input[1]; // Matrix of weights from which we can infer output dimension
      TensorProto weightsInitializer = Model.Graph.Initializer.FirstOrDefault(init => init.Name == weightsInitializerName);
      

      long outputDim = weightsInitializer.Dims[1];
      TypeProto tp = TypeProtoFromTensorProto2D(weightsInitializer, outerDim, outputDim); 

      ValueInfoProto vip = new();
      vip.Name = node.Name;
      vip.Type = tp;

      ModelProto newModel = Model.Clone();
      newModel.Graph.Output.Add(vip);
      return converter(newModel);
      //newModel.WriteToFile(modelName);
    }

    private static TypeProto TypeProtoFromTensorProto2D(TensorProto weightsInitializer, long dim1, long dim2)
    {
      TypeProto tp = new();
      tp.TensorType = new TypeProto.Types.Tensor();
      tp.TensorType.ElemType = weightsInitializer.DataType;
      tp.TensorType.Shape = ONNXHelpers.MakeTensorShape(-1, dim1, dim2);
      return tp;
    }

    public void DumpInfo() => Console.WriteLine(Model.Graph.Info());
  }

}

