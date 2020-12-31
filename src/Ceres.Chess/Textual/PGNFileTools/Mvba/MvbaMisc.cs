#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

//  * **************************************************************************
//  * Copyright (c) McCreary, Veselka, Bragg & Allen, P.C.
//  * This source code is subject to terms and conditions of the MIT License.
//  * A copy of the license can be found in the License.txt file
//  * at the root of this distribution. 
//  * By using this source code in any fashion, you are agreeing to be bound by 
//  * the terms of the MIT License.
//  * You must not remove this notice from this software.
//  * **************************************************************************

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace Ceres.Chess.Textual.PgnFileTools.MvbaCore
{
  public enum NotificationSeverity
  {
    Error,
    Warning,
    Info
  }
  public class NotificationMessage : IEquatable<NotificationMessage>
  {
    public NotificationMessage(NotificationSeverity severity,  string messageFormatString,
                               params object[] messageParameters)
    {
      Severity = severity;
      Message = String.Format(messageFormatString, messageParameters);
    }

    public NotificationMessage(NotificationSeverity severity,  string message)
    {
      Severity = severity;
      Message = message;
    }

    public string Message { get; private set; }
    public NotificationSeverity Severity { get; private set; }
    public virtual bool Equals( NotificationMessage other)
    {
      if (ReferenceEquals(null, other))
      {
        return false;
      }
      if (ReferenceEquals(this, other))
      {
        return true;
      }
      return Equals(other.Severity, Severity) && Equals(other.Message, Message);
    }

    public override bool Equals( object obj)
    {
      if (ReferenceEquals(null, obj))
      {
        return false;
      }
      if (ReferenceEquals(this, obj))
      {
        return true;
      }
      if (obj.GetType() != typeof(NotificationMessage))
      {
        return false;
      }
      return Equals((NotificationMessage)obj);
    }

  
    public override int GetHashCode()
    {
      unchecked
      {
        return Severity.GetHashCode() * 397 ^ (Message != null ? Message.GetHashCode() : 0);
      }
    }

   
    public override string ToString()
    {
      return Severity + ": " + Message;
    }
  }


  public class Notification<T> : Notification
  {
    public Notification()
    {
    }

    public Notification(NotificationMessage notificationMessage)
      : base(notificationMessage)
    {
    }

    public Notification(Notification notification, T item = default(T))
    {
      Item = item;
      Add(notification);
    }

    public new static Notification<T> Empty
    {
      get { return new Notification<T>(); }
    }

    public T Item { get; set; }

    
    public static implicit operator T(Notification<T> notification)
    {
      if (notification.HasErrors)
      {
        throw new ArgumentNullException(string.Format("Cannot implicitly cast Notification<{0}> to {0} when there are errors.", typeof(T).Name));
      }
      return notification.Item;
    }

    
    public static implicit operator Notification<T>(T item)
    {
      return new Notification<T>
      {
        Item = item
      };
    }
  }

  public abstract class NotificationBase
  {
    private readonly List<NotificationMessage> _messages;

    protected NotificationBase()
    {
      _messages = new List<NotificationMessage>();
    }

    protected NotificationBase( NotificationMessage notificationMessage)
      : this()
    {
      AddMessage(notificationMessage);
    }

    
    public string ErrorsAndWarnings
    {
      get { return !(HasErrors || HasWarnings) ? "" : GetMessages(x => x.Severity == NotificationSeverity.Error || x.Severity == NotificationSeverity.Warning); }
    }

    // ReSharper disable MemberCanBeProtected.Global
    public bool HasErrors { get; private set; }
    // ReSharper restore MemberCanBeProtected.Global

    public bool HasWarnings { get; private set; }

    
    
    
    public IEnumerable<NotificationMessage> Messages
    {
      get { return _messages.ToArray(); }
    }
    

    public void Add(Notification notification)
    {
      foreach (var message in notification.Messages)
      {
        AddMessage(message);
      }
    }

    public void Add(NotificationMessage message)
    {
      AddMessage(message);
    }

    private void AddMessage(NotificationMessage message)
    {
      if (!Messages.Any(x => x.Severity == message.Severity && x.Message == message.Message))
      {
        switch (message.Severity)
        {
          case NotificationSeverity.Error:
            HasErrors = true;
            break;
          case NotificationSeverity.Warning:
            HasWarnings = true;
            break;
        }
        _messages.Add(message);
      }
    }

    
    
    private string GetMessages( Func<NotificationMessage, bool> predicate)
    {
      // TODO: cleanup
      return "DJE disabled messages";// Messages.Where(predicate).Select(x => x.Message).Append(Environment.NewLine);
    }

    
    public override string ToString()
    {
      return ErrorsAndWarnings;
    }
  }

  public class Notification : NotificationBase
  {
    public Notification()
    {
    }

    // ReSharper disable once MemberCanBeProtected.Global
    public Notification(NotificationMessage notificationMessage)
      : base(notificationMessage)
    {
    }

    
    public static Notification Empty
    {
      get { return new Notification(); }
    }

    
    
    public static Notification ErrorFor( string messageText)
    {
      return For(NotificationSeverity.Error, messageText);
    }

       
    
    // ReSharper disable MethodOverloadWithOptionalParameter
    public static Notification ErrorFor( string messageFormatString, params object[] messageParameters)
    // ReSharper restore MethodOverloadWithOptionalParameter
    {
      return For(NotificationSeverity.Error, messageFormatString, messageParameters);
    }

    
    
    public static Notification For(NotificationSeverity severity,  string messageText)
    {
      return new Notification(new NotificationMessage(severity, messageText));
    }

    
    
    // ReSharper disable MethodOverloadWithOptionalParameter
    private static Notification For(NotificationSeverity severity,  string messageFormatString, params object[] messageParameters)
    // ReSharper restore MethodOverloadWithOptionalParameter
    {
      return new Notification(new NotificationMessage(severity, messageFormatString, messageParameters));
    }

    
    
    public static Notification InfoFor( string messageText)
    {
      return For(NotificationSeverity.Info, messageText);
    }

    
    
    // ReSharper disable MethodOverloadWithOptionalParameter
    public static Notification InfoFor( string messageFormatString, params object[] messageParameters)
    // ReSharper restore MethodOverloadWithOptionalParameter
    {
      return For(NotificationSeverity.Info, messageFormatString, messageParameters);
    }

    
    
    public static Notification WarningFor( string messageText)
    {
      return For(NotificationSeverity.Warning, messageText);
    }

    
    
    // ReSharper disable MethodOverloadWithOptionalParameter
    public static Notification WarningFor( string messageFormatString, params object[] messageParameters)
    // ReSharper restore MethodOverloadWithOptionalParameter
    {
      return For(NotificationSeverity.Warning, messageFormatString, messageParameters);
    }
  }

  public static class NotificationExtensions
  {
    public static Notification<T> ToNotification<T>( this Notification notification)
    {
      // because we can't implicitly cast up from a base class
      return new Notification<T>(notification);
    }

    public static Notification<T> ToNotification<T>( this Notification notification,  T item)
    {
      return new Notification<T>(notification, item);
    }
  }

  [Serializable]
  public class NamedConstant //DJE : INamedConstant
  {
    /// <summary>
    ///   Use Add to set
    /// </summary>
    // ReSharper disable once NotNullMemberIsNotInitialized
    public string Key { get; internal set; }

    public override string ToString()
    {
      return Key;
    }
  }

#pragma warning disable 661, 660
  [Serializable]
  public class NamedConstant<T> : NamedConstant
#pragma warning restore 661, 660
    where T : NamedConstant<T>
  {
    // ReSharper disable StaticFieldInGenericType
    private static readonly Dictionary<string, T> NamedConstants = new Dictionary<string, T>();
    // ReSharper restore StaticFieldInGenericType

    protected void Add( string key,  T item)
    {
      Key = key;
      NamedConstants.Add(key.ToLower(), item);
    }

    // ReSharper disable MemberCanBePrivate.Global
    // ReSharper disable UnusedMember.Global
    
    
    public static IEnumerable<T> GetAll()
    // ReSharper restore UnusedMember.Global
    // ReSharper restore MemberCanBePrivate.Global
    {
      EnsureValues();
      return NamedConstants.Values.Distinct();
    }

    // ReSharper disable MemberCanBePrivate.Global
    protected static T Get( string key)
    // ReSharper restore MemberCanBePrivate.Global
    {
      if (key == null)
      {
        return null;
      }
      T t;
      NamedConstants.TryGetValue(key.ToLower(), out t);
      return t;
    }

    
    public static bool operator ==( NamedConstant<T> a,  NamedConstant<T> b)
    {
      if (ReferenceEquals(a, b))
      {
        return true;
      }

      if ((object)a == null || (object)b == null)
      {
        return false;
      }

      return a.Equals(b);
    }

    
    public static bool operator !=( NamedConstant<T> a,  NamedConstant<T> b)
    {
      return !(a == b);
    }

    
    
    public static T GetFor( string key)
    {
      EnsureValues();
      return Get(key);
    }

    
    
    public static T GetDefault()
    {
      EnsureValues();
      return NamedConstantExtensions.DefaultValue<T>();
    }

    private static void EnsureValues()
    {
      if (NamedConstants.Count != 0)
      {
        return;
      }
      var fieldInfos = typeof(T).GetFields(BindingFlags.Static | BindingFlags.Public);
      // ensure its static members get created by triggering the type initializer
      fieldInfos[0].GetValue(null);
    }
  }

  public static class NamedConstantExtensions
  {
    private static readonly Dictionary<Type, object> Defaults = new Dictionary<Type, object>();
    private static readonly HashSet<Type> NoDefaults = new HashSet<Type>();

    
    
    public static T DefaultValue<T>() where T : NamedConstant<T>
    {
      var type = typeof(T);
      lock (NoDefaults)
      {
        if (NoDefaults.Contains(type))
        {
          return null;
        }
      }
      object defaultValue;
      lock (Defaults)
      {
        if (Defaults.TryGetValue(type, out defaultValue))
        {
          return (T)defaultValue;
        }
      }

      var fields = type.GetFields().Where(x => x.IsStatic);

      /*
       *    ((MemberInfo) input).CustomAttributesOfType<T>()

    internal static IEnumerable<T> CustomAttributesOfType<T>([NotNull] this MemberInfo input) where T : Attribute
		{
			return input.GetCustomAttributes(typeof(T), true).Cast<T>();
		}

       */
      throw new NotImplementedException(); //TO DO: broke this, fix if needd
      FieldInfo defaultField = default;// fields.Where(x => x.CustomAttributesOfType<DefaultKeyAttribute>().Any().FirstOrDefault();
      if (defaultField == null)
      {
        lock (NoDefaults)
        {
          if (!NoDefaults.Contains(type))
          {
            NoDefaults.Add(type);
          }
        }
        return null;
      }
      defaultValue = defaultField.GetValue(null);
      if (defaultValue == null)
      {
        lock (NoDefaults)
        {
          if (!NoDefaults.Contains(type))
          {
            NoDefaults.Add(type);
          }
        }
        return null;
      }
      lock (Defaults)
      {
        if (!Defaults.ContainsKey(type))
        {
          Defaults.Add(type, defaultValue);
        }
      }
      return (T)defaultValue;
    }

    
    
    public static T OrDefault<T>( this T value) where T : NamedConstant<T>
    {
      if (value != null)
      {
        return value;
      }
      var defaultValue = DefaultValue<T>();
      if (defaultValue == null)
      {
        throw new ArgumentException("No default value defined for Named Constant type " + typeof(T));
      }
      return defaultValue;
    }
  }

  [AttributeUsage(AttributeTargets.Field)]
  public sealed class DefaultKeyAttribute : Attribute
  {
  }

}
