﻿#region License notice

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


    public static TensorProto.Types.DataType? GetOutputDataType(ModelProto model, string nodeName, string outputName)
    {
      var node = model.Graph.Node.FirstOrDefault(n => n.Name == nodeName);
      if (node == null)
      {
        throw new Exception($"Node {nodeName} not found.");
      }

      // Check if the output is associated with a value_info entry
      var valueInfo = model.Graph.ValueInfo
          .Concat(model.Graph.Input)
          .Concat(model.Graph.Output)
          .FirstOrDefault(info => info.Name == outputName);

      if (valueInfo != null)
      {
        // Get the data type from the type field (TensorTypeProto)
        return (TensorProto.Types.DataType)valueInfo.Type?.TensorType?.ElemType;
      }

      // If not found in value_info, check if it's in the initializer (constant tensor)
      var initializer = model.Graph.Initializer.FirstOrDefault(init => init.Name == outputName);
      if (initializer != null)
      {
        return (TensorProto.Types.DataType)initializer.DataType;
      }

      // If not found, return null or handle accordingly
      return null;
    }
    
  public List<long> GetOutputShape(ModelProto model, string nodeName, string outputName)
    {
      // Look up the node in the graph
      var node = model.Graph.Node.FirstOrDefault(n => n.Name == nodeName);
      if (node == null)
      {
        throw new Exception($"Node {nodeName} not found.");
      }

      // Check if the output is associated with a value_info entry
      ValueInfoProto valueInfo = model.Graph.ValueInfo
          .Concat(model.Graph.Input)
          .Concat(model.Graph.Output)
          .FirstOrDefault(info => info.Name == outputName);

      if (valueInfo != null)
      {
        // Get the shape from the type field (TensorTypeProto)
        var shapeProto = valueInfo.Type?.TensorType?.Shape;
        if (shapeProto != null)
        {
          return shapeProto.Dim.Select(dim => dim.DimValue).ToList();
        }
      }

      // If not found in value_info, check if it's in the initializer (constant tensor)
      var initializer = model.Graph.Initializer.FirstOrDefault(init => init.Name == outputName);
      if (initializer != null)
      {
        return initializer.Dims.ToList();
      }

      // If shape cannot be determined, return an empty list or null
      return null;
  }



    /// <summary>
    /// Returns a new model with specified new outputs (from certain of the inner nodes) added.
    /// </summary>
    /// <param name="includeNodePredicate"></param>
    /// <returns></returns>
    public ModelProto WithAddedOutputNodes(Predicate<NodeProto> includeNodePredicate)
      => WithAddedOutputNodes(Model.Graph.Node.Where(p => p.Output != null
                                                       && includeNodePredicate(p)
                                                       && p.Output.Count > 0).Select(p => p.Name));


    /// <summary>
    /// Returns a new model with specified new outputs (from certain of the inner nodes) added.
    /// </summary>
    /// <param name="nodeNames"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public ModelProto WithAddedOutputNodes(IEnumerable<string> nodeNames)
    {
      ModelProto newModel = Model.Clone();

      foreach (string nodeName in nodeNames)
      {
        NodeProto node = NodeWithName(nodeName);
        if (node == null)
        {
          throw new Exception("Node not found: " + nodeName);
        }

        for (int i=0;i<node.Output.Count;i++)
        {
          //string weightsInitializerName = node.Input[1]; // Matrix of weights from which we can infer output dimension
          //TensorProto weightsInitializer = Model.Graph.Initializer.FirstOrDefault(init => init.Name == weightsInitializerName);     
          //long outputDim = weightsInitializer.Dims[1];

          List<long> thisShape = GetOutputShape(newModel, nodeName, node.Output[i]);
          if (thisShape == null)
          {
            throw new Exception($"Could not infer shape for output {node.Output[i]} of node {nodeName}");
          }

          TensorProto.Types.DataType? thisDataType = GetOutputDataType(newModel, nodeName, node.Output[i]);
          if (thisDataType == null)
          {
            throw new Exception($"Could not infer data type for output {node.Output[i]} of node {nodeName}");
          }

          TypeProto tp = new();
          tp.TensorType = new TypeProto.Types.Tensor();
          tp.TensorType.ElemType = (int)thisDataType;
//          tp.TensorType.Shape = ONNXHelpers.MakeTensorShape(thisShape.ToArray());

          ValueInfoProto vip = new();
          vip.Name = node.Output[i];
          vip.Type = tp;

          newModel.Graph.Output.Add(vip);
        }
      }

      return newModel;
    }


    /// <summary>
    /// Returns a new model with a specified new output (from one of the inner nodes) added.
    /// </summary>
    /// <param name="nodeName"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public ModelProto WithAddedOutputNode(string nodeName)
      => WithAddedOutputNodes(new string[] { nodeName }); 
    


    private static TypeProto TypeProtoFromTensorProto2D(TensorProto weightsInitializer, params long[] dims)
    {
      TypeProto tp = new();
      tp.TensorType = new TypeProto.Types.Tensor();
      tp.TensorType.ElemType = weightsInitializer.DataType;
      tp.TensorType.Shape = ONNXHelpers.MakeTensorShape(dims);
      return tp;
    }


    public void DumpInfo() => Console.WriteLine(Model.Graph.Info());
  }

}

